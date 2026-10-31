#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "core/device.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <variant>

namespace ninfer::models::qwen3_5::detail {

bool ProgramImpl::valid_sequence(SequenceHandle handle) const noexcept {
    if (ContractAccess::owner(handle) != this) { return false; }
    const std::uint32_t lane = ContractAccess::lane(handle).value;
    if (lane >= max_concurrency || ContractAccess::epoch(handle) != lane_epochs[lane]) {
        return false;
    }
    if (active_continuations[lane] >= continuation_capacity ||
        continuation_slots[active_continuations[lane]].role != ContinuationSlotRole::Active) {
        return false;
    }
    const Lifecycle lifecycle = requests[lane].lifecycle;
    return lifecycle == Lifecycle::Prefilling || lifecycle == Lifecycle::Active ||
           lifecycle == Lifecycle::Pending || lifecycle == Lifecycle::Finishable;
}

bool ProgramImpl::valid_continuation(const ContinuationHandle& handle) const noexcept {
    if (ContractAccess::owner(handle) != this) { return false; }
    const std::uint32_t index = ContractAccess::index(handle);
    return index < continuation_capacity &&
           ContractAccess::epoch(handle) == continuation_slots[index].generation &&
           continuation_slots[index].role == ContinuationSlotRole::Catalogued;
}

bool ProgramImpl::valid_shared_prefix(const SharedPrefixHandle& handle) const noexcept {
    if (ContractAccess::owner(handle) != this) { return false; }
    const std::uint32_t index = ContractAccess::index(handle);
    return index < shared_prefix_capacity &&
           ContractAccess::epoch(handle) == shared_prefix_slots[index].generation &&
           shared_prefix_slots[index].role == SharedPrefixSlotRole::Catalogued;
}

bool ProgramImpl::valid_capture_offer(const CaptureOffer& offer) const noexcept {
    if (ContractAccess::owner(offer) != this) { return false; }
    const std::uint32_t lane = ContractAccess::lane(offer).value;
    if (lane >= max_concurrency || ContractAccess::epoch(offer) != lane_epochs[lane] ||
        (requests[lane].lifecycle != Lifecycle::Prefilling &&
         requests[lane].lifecycle != Lifecycle::Active) ||
        !requests[lane].prefill) {
        return false;
    }
    const RequestControl::Prefill& prefill = *requests[lane].prefill;
    return prefill.pending_capture_offer != 0 &&
           prefill.pending_capture_offer == ContractAccess::id(offer) &&
           prefill.next_capture < prefill.capture_groups.size() &&
           prefill.cursor == prefill.capture_groups[prefill.next_capture].frontier;
}

bool ProgramImpl::materialization_pins(std::uint32_t index,
                                       std::uint64_t generation) const noexcept {
    const MaterializationTransaction* transaction_ptr =
        std::get_if<MaterializationTransaction>(&context_transaction_);
    if (transaction_ptr == nullptr) { return false; }
    const MaterializationTransaction& transaction = *transaction_ptr;
    if (transaction.has_source && transaction.source_index == index &&
        transaction.source_generation == generation) {
        return true;
    }
    for (std::size_t victim = 0; victim < transaction.victim_count; ++victim) {
        if (!transaction.victim_released[victim] && transaction.victim_indices[victim] == index &&
            transaction.victim_generations[victim] == generation) {
            return true;
        }
    }
    return false;
}

bool ProgramImpl::valid_pending(const PendingBatch& pending) const noexcept {
    if (ContractAccess::owner(pending) != this || !pending_transaction_ ||
        ContractAccess::transaction(pending) != pending_transaction_->id) {
        return false;
    }
    const auto rows = ContractAccess::rows(pending);
    if (rows.size() != pending_transaction_->size) { return false; }
    for (std::size_t row = 0; row < rows.size(); ++row) {
        if (!valid_sequence(rows[row]) ||
            ContractAccess::lane(rows[row]).value != pending_transaction_->lanes[row] ||
            ContractAccess::epoch(rows[row]) != pending_transaction_->epochs[row] ||
            requests[pending_transaction_->lanes[row]].lifecycle != Lifecycle::Pending) {
            return false;
        }
    }
    return true;
}

void ProgramImpl::invalidate_lane(std::uint32_t lane) noexcept {
    if (lane >= max_concurrency) { return; }
    ++lane_epochs[lane];
    if (lane_epochs[lane] == 0) { ++lane_epochs[lane]; }
}

SequenceState& ProgramImpl::active_sequence(std::uint32_t lane) {
    if (lane >= max_concurrency) { throw std::out_of_range("active lane is out of range"); }
    const std::uint32_t index = active_continuations[lane];
    if (index >= continuation_capacity ||
        continuation_slots[index].role != ContinuationSlotRole::Active) {
        throw std::logic_error("active lane has no continuation binding");
    }
    return continuation_states[index];
}

const SequenceState& ProgramImpl::active_sequence(std::uint32_t lane) const {
    if (lane >= max_concurrency) { throw std::out_of_range("active lane is out of range"); }
    const std::uint32_t index = active_continuations[lane];
    if (index >= continuation_capacity ||
        continuation_slots[index].role != ContinuationSlotRole::Active) {
        throw std::logic_error("active lane has no continuation binding");
    }
    return continuation_states[index];
}

std::optional<std::uint32_t> ProgramImpl::allocate_continuation_slot() noexcept {
    for (std::uint32_t index = 0; index < continuation_capacity; ++index) {
        if (continuation_slots[index].role == ContinuationSlotRole::Free) {
            continuation_slots[index].role = ContinuationSlotRole::Active;
            return index;
        }
    }
    return std::nullopt;
}

bool ProgramImpl::can_release_continuation_slot_strict(std::uint32_t index) const {
    if (index >= continuation_capacity || !state_store || !text_kv_addresses || !text_kv_pages ||
        continuation_slots[index].role != ContinuationSlotRole::Catalogued) {
        return false;
    }
    const SequenceState& sequence = continuation_states[index];
    if (sequence.state.fork_pending || !sequence.shared_prefix_references.empty() || !sequence.kv ||
        !text_kv_addresses->can_release(sequence.kv->text)) {
        return false;
    }
    if (sequence.kv->backend) {
        if (!backend_kv_addresses || !backend_kv_pages ||
            !backend_kv_addresses->can_release(*sequence.kv->backend)) {
            return false;
        }
    }

    const auto validate_state = [&](StateImageHandle handle, bool release_object) {
        if (!state_store->valid(handle)) { return false; }
        const std::uint32_t owned = owned_checkpoint_references(sequence, handle);
        const std::uint32_t total = state_store->checkpoint_references(handle);
        if (owned > total ||
            (owned != 0 && state_store->role(handle) != StateImageRole::CheckpointImmutable)) {
            return false;
        }
        return !release_object || total != owned ||
               state_store->can_release_after_checkpoint_references(handle, owned);
    };
    const auto repeated_before_anchor = [&](std::size_t anchor_index, StateImageHandle handle) {
        if (handle == sequence.state.read || handle == sequence.state.write ||
            (sequence.rewrite_state && handle == *sequence.rewrite_state)) {
            return true;
        }
        for (std::size_t prior = 0; prior < anchor_index; ++prior) {
            if (sequence.long_anchors[prior].state == handle) { return true; }
        }
        return false;
    };

    if (sequence.endpoint_valid) {
        if (!validate_state(sequence.state.read, !sequence.state.read_has_external_owner() ||
                                                     sequence.state.read == sequence.state.write)) {
            return false;
        }
        if (sequence.state.write != sequence.state.read &&
            !validate_state(sequence.state.write, true)) {
            return false;
        }
    } else if (state_store->valid(sequence.state.read) ||
               state_store->valid(sequence.state.write) || sequence.state.borrows_read()) {
        return false;
    }
    if (sequence.rewrite_state && *sequence.rewrite_state != sequence.state.read &&
        *sequence.rewrite_state != sequence.state.write &&
        !validate_state(*sequence.rewrite_state, true)) {
        return false;
    }
    for (std::size_t anchor = 0; anchor < sequence.long_anchors.size(); ++anchor) {
        const StateImageHandle handle = sequence.long_anchors[anchor].state;
        if (!repeated_before_anchor(anchor, handle) && !validate_state(handle, true)) {
            return false;
        }
    }
    if (sequence.reserved_state) {
        const StateImageHandle handle = *sequence.reserved_state;
        bool repeated = handle == sequence.state.read || handle == sequence.state.write ||
                        (sequence.rewrite_state && handle == *sequence.rewrite_state);
        for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
            repeated = repeated || anchor.state == handle;
        }
        if (!repeated && !validate_state(handle, true)) { return false; }
    }
    return true;
}

void ProgramImpl::release_continuation_slot_strict(std::uint32_t index) noexcept {
    try {
        if (!can_release_continuation_slot_strict(index)) { std::terminate(); }
    } catch (...) { std::terminate(); }
    SequenceState& sequence = continuation_states[index];
    release_sequence_kv_strict(sequence);
    release_sequence_state_strict(sequence);
    retire_continuation_slot(index);
}

void ProgramImpl::release_continuation_slot_best_effort(std::uint32_t index) noexcept {
    if (index >= continuation_capacity ||
        continuation_slots[index].role == ContinuationSlotRole::Free) {
        return;
    }
    SequenceState& sequence = continuation_states[index];
    release_active_shared_references(sequence);
    release_sequence_kv(sequence);
    release_sequence_state(sequence);
    retire_continuation_slot(index);
}

void ProgramImpl::retire_continuation_slot(std::uint32_t index) noexcept {
    if (index >= continuation_capacity) { std::terminate(); }
    SequenceState& sequence     = continuation_states[index];
    sequence.execution_frontier = 0;
    sequence.ledger_frontier    = 0;
    sequence.ledger.clear();
    sequence.ngram_observed = 0;
    sequence.prefix_identity.clear();
    sequence.prefix_digests.clear();
    sequence.rope_delta              = 0;
    sequence.text_kv_valid           = 0;
    sequence.mtp_kv_valid            = 0;
    sequence.dflash_context_frontier = 0;
    sequence.mtp_draft_count         = 0;
    sequence.tail_hidden_valid       = false;
    sequence.endpoint_valid          = false;
    sequence.rewrite_checkpoint      = {};
    sequence.rebuild_work            = {};
    sequence.rebuild_tail_begin      = 0;
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        if (active_continuations[lane] == index) {
            active_continuations[lane] = continuation_capacity;
        }
    }
    ContinuationSlot& slot = continuation_slots[index];
    slot.role              = ContinuationSlotRole::Free;
    if (++slot.generation == 0) { ++slot.generation; }
}

detail::PhysicalResources
ProgramImpl::sequence_exclusive_state_resources(const SequenceState& sequence) const {
    if (!state_store) {
        throw std::logic_error("sequence StateImage resources have no physical store");
    }
    detail::PhysicalResources out;
    std::array<StateImageHandle, 4> states{};
    std::uint32_t state_count = 0;
    const auto add_state      = [&](StateImageHandle handle) {
        if (!state_store->valid(handle)) {
            throw std::logic_error("sequence owner has a stale StateImage");
        }
        if (!state_exclusive_to_sequence(sequence, handle)) { return; }
        for (std::uint32_t index = 0; index < state_count; ++index) {
            if (states[index] == handle) { return; }
        }
        states[state_count++]                 = handle;
        const StateReplicaResidency residency = state_store->residency(handle);
        if (residency == StateReplicaResidency::DeviceOnly ||
            residency == StateReplicaResidency::Both) {
            ++out.device.state_slots;
        }
        if (residency == StateReplicaResidency::HostOnly ||
            residency == StateReplicaResidency::Both) {
            ++out.host.state_slots;
        }
    };
    const bool has_read_state  = sequence.state.read.valid();
    const bool has_write_state = sequence.state.write.valid();
    if (has_read_state != has_write_state) {
        throw std::logic_error("sequence owner has a partial primary StateImage pair");
    }
    if (sequence.state.borrows_read() &&
        (!sequence.state.fork_pending || sequence.state.read == sequence.state.write)) {
        throw std::logic_error("sequence has an invalid borrowed StateImage source");
    }
    if (has_read_state) {
        if (!sequence.state.borrows_read() || sequence.state.read == sequence.state.write) {
            add_state(sequence.state.read);
        }
        add_state(sequence.state.write);
    }
    if (sequence.rewrite_state) { add_state(*sequence.rewrite_state); }
    if (sequence.reserved_state) { add_state(*sequence.reserved_state); }
    for (std::size_t anchor_index = 0; anchor_index < sequence.long_anchors.size();
         ++anchor_index) {
        const StateImageHandle handle = sequence.long_anchors[anchor_index].state;
        if (!state_store->valid(handle)) {
            throw std::logic_error("sequence owner has a stale long-anchor StateImage");
        }
        if (!state_exclusive_to_sequence(sequence, handle)) { continue; }
        bool seen = false;
        for (std::uint32_t index = 0; index < std::min<std::uint32_t>(state_count, states.size());
             ++index) {
            if (states[index] == handle) { seen = true; }
        }
        for (std::size_t prior = 0; !seen && prior < anchor_index; ++prior) {
            if (sequence.long_anchors[prior].state == handle) { seen = true; }
        }
        if (seen) { continue; }
        const StateReplicaResidency residency = state_store->residency(handle);
        if (residency == StateReplicaResidency::DeviceOnly ||
            residency == StateReplicaResidency::Both) {
            ++out.device.state_slots;
        }
        if (residency == StateReplicaResidency::HostOnly ||
            residency == StateReplicaResidency::Both) {
            ++out.host.state_slots;
        }
    }
    return out;
}

detail::PhysicalResources
ProgramImpl::owner_exclusive_resources(const SequenceState& sequence) const {
    if (!state_store || !text_kv_addresses || !text_kv_pages) {
        throw std::logic_error("sequence owner resources have no physical stores");
    }
    detail::PhysicalResources out = sequence_exclusive_state_resources(sequence);

    {
        if (!sequence.kv) { throw std::logic_error("sequence owner has no KV address bundle"); }
        const auto add_kv = [&](const KVAddressSpaceStore& addresses,
                                const LogicalKVPageStore& pages, KVAddressSpaceHandle address,
                                std::uint32_t& device_pages) {
            if (!addresses.valid(address)) { throw std::logic_error("stale KV address space"); }
            for (std::uint32_t page = 0; page < addresses.mapped_pages(address); ++page) {
                const LogicalKVPageHandle logical = addresses.logical_page(address, page);
                // A shared logical page contributes to aggregate occupancy once. Releasing this
                // address cannot free either replica while another address still references it,
                // so it is not part of this owner's exact transition effect.
                if (pages.address_references(logical) > 1) { continue; }
                if (pages.device_resident(logical)) { ++device_pages; }
                if (pages.host_resident(logical)) {
                    if (!host_kv_extents) {
                        throw std::logic_error("missing Host KV extent store");
                    }
                    const HostKVPageReplica& replica = pages.host_replica(logical);
                    const std::size_t stride =
                        host_kv_extents->view(replica.extent).layout().page_stride;
                    if (stride > std::numeric_limits<std::size_t>::max() - out.host.kv_bytes) {
                        throw std::overflow_error("resident Host KV byte count overflow");
                    }
                    out.host.kv_bytes += stride;
                }
            }
            if (addresses.active(address)) {
                // Physical, not logical: an active address is charged for the Device pages it holds
                // plus the pool claim on its unmapped tail (kvmem-port.md 2.10). Under windowing the
                // claim is capped by the pool, so entitlement - mapped would over-report occupancy
                // and make eviction effects claim more Device memory than exists.
                const std::uint32_t tail =
                    addresses.claimed_tail_pages(address);
                if (tail > std::numeric_limits<std::uint32_t>::max() - device_pages) {
                    throw std::overflow_error("owner active KV claim overflow");
                }
                device_pages += tail;
            }
        };
        add_kv(*text_kv_addresses, *text_kv_pages, sequence.kv->text, out.device.main_kv_pages);
        if (sequence.kv->backend) {
            if (!backend_kv_addresses || !backend_kv_pages) {
                throw std::logic_error("missing Backend KV stores");
            }
            add_kv(*backend_kv_addresses, *backend_kv_pages, *sequence.kv->backend,
                   out.device.backend_kv_pages);
        }
    }
    return out;
}

detail::PhysicalResources
ProgramImpl::owner_exclusive_resources(const SharedPrefixState& shared) const {
    if (!state_store || !text_kv_addresses || !text_kv_pages) {
        throw std::logic_error("shared owner resources have no physical stores");
    }
    detail::PhysicalResources out;
    {
        if (!shared.kv || !shared.identity || !state_store->valid(shared.state)) {
            throw std::logic_error("shared prefix has incomplete resident physical state");
        }
        if (state_store->checkpoint_references(shared.state) == 0) {
            throw std::logic_error("shared prefix StateImage has no checkpoint reference");
        }
        const StateReplicaResidency residency = state_store->residency(shared.state);
        if (state_store->checkpoint_references(shared.state) == 1) {
            if (residency == StateReplicaResidency::DeviceOnly ||
                residency == StateReplicaResidency::Both) {
                ++out.device.state_slots;
            }
            if (residency == StateReplicaResidency::HostOnly ||
                residency == StateReplicaResidency::Both) {
                ++out.host.state_slots;
            }
        }
        const auto add_kv = [&](const KVAddressSpaceStore& addresses,
                                const LogicalKVPageStore& pages, KVAddressSpaceHandle address,
                                std::uint32_t& device_pages) {
            if (!addresses.valid(address)) { throw std::logic_error("stale shared KV address"); }
            for (std::uint32_t page = 0; page < addresses.mapped_pages(address); ++page) {
                const LogicalKVPageHandle logical = addresses.logical_page(address, page);
                if (pages.address_references(logical) != 1) { continue; }
                if (pages.device_resident(logical)) { ++device_pages; }
                if (pages.host_resident(logical)) {
                    if (!host_kv_extents) {
                        throw std::logic_error("missing Host KV extent store");
                    }
                    const HostKVPageReplica& replica = pages.host_replica(logical);
                    const std::size_t stride =
                        host_kv_extents->view(replica.extent).layout().page_stride;
                    if (stride > std::numeric_limits<std::size_t>::max() - out.host.kv_bytes) {
                        throw std::overflow_error("shared Host KV byte count overflow");
                    }
                    out.host.kv_bytes += stride;
                }
            }
        };
        add_kv(*text_kv_addresses, *text_kv_pages, shared.kv->text, out.device.main_kv_pages);
        if (shared.kv->backend) {
            if (!backend_kv_addresses || !backend_kv_pages) {
                throw std::logic_error("missing shared Backend KV stores");
            }
            add_kv(*backend_kv_addresses, *backend_kv_pages, *shared.kv->backend,
                   out.device.backend_kv_pages);
        }
    }
    return out;
}

detail::PhysicalResources ProgramImpl::physical_occupancy() const noexcept {
    detail::PhysicalResources out;
    for (const RequestControl& request : requests) {
        if (request.lifecycle != Lifecycle::Empty) { ++out.device.active_lanes; }
    }
    if (state_store) {
        out.device.state_slots = state_store->device_occupied();
        out.host.state_slots   = state_store->host_occupied();
    }
    if (text_kv_pages) {
        const DeviceKVPagePool& pool = text_kv_pages->physical_pool();
        out.device.main_kv_pages     = pool.allocated_pages() + pool.reserved_pages();
    }
    if (backend_kv_pages) {
        const DeviceKVPagePool& pool = backend_kv_pages->physical_pool();
        out.device.backend_kv_pages  = pool.allocated_pages() + pool.reserved_pages();
    }
    if (host_kv_arena) { out.host.kv_bytes = host_kv_arena->occupied_bytes(); }
    return out;
}

detail::PhysicalResources
ProgramImpl::materialization_deficit(const ResourceCandidateState& admission) const {
    // Pressure is relative to this candidate's real peak. Treating every dimension as scarce
    // would forbid Device-to-Host demotion even when Host capacity is available.
    const detail::PhysicalResources required =
        checked_resource_sum(physical_occupancy(), admission.demand.physical_peak_additional);
    return positive_resource_difference(required, admission_capacity());
}

detail::PhysicalResources
ProgramImpl::guided_materialization_deficit(const ResourceCandidateState& admission,
                                            const detail::PhysicalDelta& pressure) const {
    // Pressure acts on the candidate's complete peak, not on its already-clamped deficit.  Applying
    // a demotion directly to a zero Host deficit would otherwise manufacture Host pressure even
    // when the arena has ample slack and steer the heuristic toward unnecessary destruction.
    const detail::PhysicalResources projected_peak = positive_resource_difference(
        checked_resource_sum(admission.demand.physical_peak_additional, pressure.added),
        pressure.removed);
    const detail::PhysicalResources required =
        checked_resource_sum(physical_occupancy(), projected_peak);
    return positive_resource_difference(required, admission_capacity());
}

bool ProgramImpl::physical_peak_fits(detail::PhysicalResources peak) const noexcept {
    const detail::PhysicalResources occupied = physical_occupancy();
    const detail::PhysicalResources limits   = admission_capacity();
    const auto fits_u32 = [](std::uint32_t used, std::uint32_t added, std::uint32_t capacity) {
        return added <= capacity && used <= capacity - added;
    };
    const auto fits_size = [](std::size_t used, std::size_t added, std::size_t capacity) {
        return added <= capacity && used <= capacity - added;
    };
    return fits_u32(occupied.device.active_lanes, peak.device.active_lanes,
                    limits.device.active_lanes) &&
           fits_u32(occupied.device.state_slots, peak.device.state_slots,
                    limits.device.state_slots) &&
           fits_u32(occupied.device.main_kv_pages, peak.device.main_kv_pages,
                    limits.device.main_kv_pages) &&
           fits_u32(occupied.device.backend_kv_pages, peak.device.backend_kv_pages,
                    limits.device.backend_kv_pages) &&
           fits_u32(occupied.host.state_slots, peak.host.state_slots, limits.host.state_slots) &&
           fits_size(occupied.host.kv_bytes, peak.host.kv_bytes, limits.host.kv_bytes);
}

// Every way this resolution fails is a stale plan rather than a broken Program, so preparation
// consults it directly and aborts its transaction. selected_state keeps throwing for the callers
// that have already established the path is live.
std::optional<StateImageHandle>
ProgramImpl::try_selected_state(const SequenceState& sequence, ReusePath reuse,
                                std::optional<runtime::CheckpointRef> checkpoint) const {
    if (reuse == ReusePath::PrivateEndpoint) {
        if (!sequence.endpoint_valid || !state_store->valid(sequence.state.read)) {
            return std::nullopt;
        }
        return sequence.state.read;
    }
    if (is_rewrite_checkpoint_restore(reuse) && sequence.rewrite_state &&
        state_store->valid(*sequence.rewrite_state)) {
        return *sequence.rewrite_state;
    }
    if (reuse == ReusePath::PrivateLongAnchor) {
        if (!checkpoint || checkpoint->kind != runtime::CheckpointKind::LongAnchor) {
            return std::nullopt;
        }
        const auto anchor = std::find_if(sequence.long_anchors.begin(), sequence.long_anchors.end(),
                                         [&](const LongAnchorCheckpoint& candidate) {
                                             return candidate.frontier == checkpoint->frontier &&
                                                    candidate.ordinal == checkpoint->ordinal;
                                         });
        if (anchor != sequence.long_anchors.end() && state_store->valid(anchor->state)) {
            return anchor->state;
        }
    }
    return std::nullopt;
}

StateImageHandle
ProgramImpl::selected_state(const SequenceState& sequence, ReusePath reuse,
                            std::optional<runtime::CheckpointRef> checkpoint) const {
    const std::optional<StateImageHandle> selected =
        try_selected_state(sequence, reuse, std::move(checkpoint));
    if (!selected) { throw std::logic_error("materialization path has no selected StateImage"); }
    return *selected;
}

std::uint32_t
ProgramImpl::selected_state_consumed_references(const SequenceState& sequence, ReusePath reuse,
                                                RewriteCheckpointDisposition rewrite_disposition,
                                                std::optional<runtime::CheckpointRef> checkpoint,
                                                std::uint32_t reuse_base) const {
    const StateImageHandle selected   = selected_state(sequence, reuse, checkpoint);
    std::uint32_t consumed_references = 0;
    if (is_rewrite_checkpoint_restore(reuse) &&
        rewrite_disposition != RewriteCheckpointDisposition::RetainExisting) {
        if (!sequence.rewrite_state || *sequence.rewrite_state != selected) {
            throw std::logic_error("selected rewrite StateImage is unavailable");
        }
        consumed_references = 1;
    } else if (reuse == ReusePath::PrivateEndpoint &&
               rewrite_disposition != RewriteCheckpointDisposition::RetainExisting &&
               sequence.rewrite_state && *sequence.rewrite_state == selected) {
        consumed_references = 1;
    }
    for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
        if (anchor.frontier > reuse_base && anchor.state == selected) {
            if (consumed_references == std::numeric_limits<std::uint32_t>::max()) {
                throw std::overflow_error("consumed StateImage reference inventory overflow");
            }
            ++consumed_references;
        }
    }
    const std::uint32_t references = state_store->checkpoint_references(selected);
    if (consumed_references > references) {
        throw std::logic_error("selected StateImage reference inventory is inconsistent");
    }
    return consumed_references;
}

bool ProgramImpl::selected_state_requires_fork(const SequenceState& sequence, ReusePath reuse,
                                               RewriteCheckpointDisposition rewrite_disposition,
                                               std::optional<runtime::CheckpointRef> checkpoint,
                                               std::uint32_t reuse_base) const {
    const StateImageHandle selected = selected_state(sequence, reuse, checkpoint);
    return state_store->checkpoint_references(selected) !=
           selected_state_consumed_references(sequence, reuse, rewrite_disposition, checkpoint,
                                              reuse_base);
}

bool ProgramImpl::can_retain_rewrite_checkpoint(const PreparedPromptData& prompt,
                                                const RewriteCheckpointSpec& desired,
                                                const SequenceState& sequence, ReusePath reuse,
                                                std::uint32_t reuse_base) const {
    if (!sequence.rewrite_checkpoint.valid || !sequence.rewrite_state ||
        !state_store->valid(*sequence.rewrite_state) ||
        !qwen3_5::detail::prefix_matches(prompt, sequence.ledger, sequence.prefix_identity,
                                         sequence.rewrite_checkpoint.frontier)) {
        return false;
    }
    if (sequence.rewrite_checkpoint.frontier == desired.frontier) { return true; }
    return is_rewrite_checkpoint_restore(reuse) &&
           sequence.rewrite_checkpoint.frontier == reuse_base && desired.frontier <= reuse_base;
}

std::uint32_t ProgramImpl::device_kv_prefix_pages(const KVAddressSpaceStore& addresses,
                                                  KVAddressSpaceHandle address,
                                                  std::uint32_t frontier) const {
    const std::uint32_t required = kv_pages_for_frontier(frontier);
    if (required > addresses.mapped_pages(address)) {
        throw std::logic_error("checkpoint KV requirement exceeds address membership");
    }
    const LogicalKVPageStore& pages =
        (&addresses == text_kv_addresses.get()) ? *text_kv_pages : *backend_kv_pages;
    std::uint32_t resident = 0;
    for (std::uint32_t page = 0; page < required; ++page) {
        if (pages.device_resident(addresses.logical_page(address, page))) { ++resident; }
    }
    return resident;
}

std::uint32_t ProgramImpl::shared_kv_prefix_pages(const KVAddressSpaceStore& addresses,
                                                  KVAddressSpaceHandle address,
                                                  std::uint32_t frontier) const {
    const std::uint32_t required = kv_pages_for_frontier(frontier);
    if (required > addresses.mapped_pages(address)) {
        throw std::logic_error("checkpoint KV requirement exceeds address membership");
    }
    const LogicalKVPageStore& pages =
        (&addresses == text_kv_addresses.get()) ? *text_kv_pages : *backend_kv_pages;
    std::uint32_t shared = 0;
    for (std::uint32_t page = 0; page < required; ++page) {
        if (pages.address_references(addresses.logical_page(address, page)) <= 1) { continue; }
        if (page + 1U == required && frontier % static_cast<std::uint32_t>(kPagedKVPageSize) != 0) {
            continue;
        }
        ++shared;
    }
    return shared;
}

std::uint32_t ProgramImpl::shared_device_kv_prefix_pages(const KVAddressSpaceStore& addresses,
                                                         KVAddressSpaceHandle address,
                                                         std::uint32_t frontier) const {
    const std::uint32_t required = kv_pages_for_frontier(frontier);
    if (required > addresses.mapped_pages(address)) {
        throw std::logic_error("checkpoint KV requirement exceeds address membership");
    }
    const LogicalKVPageStore& pages =
        (&addresses == text_kv_addresses.get()) ? *text_kv_pages : *backend_kv_pages;
    std::uint32_t resident = 0;
    for (std::uint32_t page = 0; page < required; ++page) {
        const LogicalKVPageHandle logical = addresses.logical_page(address, page);
        if (pages.address_references(logical) > 1 && pages.device_resident(logical)) { ++resident; }
    }
    return resident;
}

bool ProgramImpl::partial_tail_cow_required(const KVAddressSpaceStore& addresses,
                                            KVAddressSpaceHandle address,
                                            std::uint32_t frontier) const {
    if (frontier == 0 || frontier % static_cast<std::uint32_t>(kPagedKVPageSize) == 0) {
        return false;
    }
    const std::uint32_t required = kv_pages_for_frontier(frontier);
    if (required > addresses.mapped_pages(address)) {
        throw std::logic_error("checkpoint KV requirement exceeds address membership");
    }
    const LogicalKVPageStore& pages =
        (&addresses == text_kv_addresses.get()) ? *text_kv_pages : *backend_kv_pages;
    const LogicalKVPageHandle tail = addresses.logical_page(address, required - 1U);
    return pages.address_references(tail) > 1 || !pages.device_resident(tail);
}

std::uint32_t
ProgramImpl::missing_shared_device_kv_prefix_pages(const KVAddressSpaceStore& addresses,
                                                   KVAddressSpaceHandle address,
                                                   std::uint32_t frontier) const {
    const std::uint32_t required = kv_pages_for_frontier(frontier);
    if (required > addresses.mapped_pages(address)) {
        throw std::logic_error("checkpoint KV requirement exceeds address membership");
    }
    const LogicalKVPageStore& pages =
        (&addresses == text_kv_addresses.get()) ? *text_kv_pages : *backend_kv_pages;
    std::uint32_t missing = 0;
    for (std::uint32_t page = 0; page < required; ++page) {
        const LogicalKVPageHandle logical = addresses.logical_page(address, page);
        if (pages.address_references(logical) > 1 && !pages.device_resident(logical)) { ++missing; }
    }
    return missing;
}

std::size_t ProgramImpl::host_kv_prefix_bytes(const KVAddressSpaceStore& addresses,
                                              KVAddressSpaceHandle address,
                                              std::uint32_t frontier) const noexcept {
    if (!host_kv_extents) { return 0; }
    try {
        const LogicalKVPageStore& pages =
            (&addresses == text_kv_addresses.get()) ? *text_kv_pages : *backend_kv_pages;
        const std::uint32_t required_pages = kv_pages_for_frontier(frontier);
        if (required_pages > addresses.mapped_pages(address)) { return 0; }
        std::size_t bytes = 0;
        for (std::uint32_t page = 0; page < required_pages; ++page) {
            const LogicalKVPageHandle logical = addresses.logical_page(address, page);
            if (pages.address_references(logical) > 1) { continue; }
            if (!pages.host_resident(logical)) { continue; }
            if (page + 1U == required_pages &&
                frontier % static_cast<std::uint32_t>(kPagedKVPageSize) != 0 &&
                partial_tail_cow_required(addresses, address, frontier)) {
                continue;
            }
            const std::uint32_t begin = page * static_cast<std::uint32_t>(kPagedKVPageSize);
            const std::uint32_t selected_columns =
                std::min(static_cast<std::uint32_t>(kPagedKVPageSize), frontier - begin);
            if (selected_columns != pages.committed_columns(logical)) {
                // A destructive private rewrite changes this tail page's content epoch, so its
                // old Host replica cannot remain part of the active entitlement.
                continue;
            }
            const std::size_t stride =
                host_kv_extents->view(pages.host_replica(logical).extent).layout().page_stride;
            if (stride > std::numeric_limits<std::size_t>::max() - bytes) { return 0; }
            bytes += stride;
        }
        return bytes;
    } catch (...) { return 0; }
}

std::pair<std::uint32_t, std::uint32_t>
ProgramImpl::tiered_kv_claim(const SequenceKVBundle* kv, std::uint32_t main_pages) const {
    if (kv == nullptr || !text_kv_addresses || !text_kv_pages ||
        !text_kv_addresses->valid(kv->text)) {
        // Unbound or released KV: restore materializes the whole prefix on Device, so the
        // claim is entirely resident - there is no window-managed backing to report.
        return {main_pages, 0U};
    }
    // Window口径 only: the store's demoted flags are the host_only half of the claim
    // (kv_store.h demoted_pages). Pressure may additionally evict parked Device replicas
    // without touching those flags; that pre-existing pool behavior is not tracked here.
    const std::uint32_t mapped = text_kv_addresses->mapped_pages(kv->text);
    std::uint32_t host_only    = 0;
    for (std::uint32_t page = 0; page < std::min(main_pages, mapped); ++page) {
        if (text_kv_addresses->demoted_page(kv->text, page)) { ++host_only; }
    }
    return {main_pages - host_only, host_only};
}

qwen3_5::CheckpointSummary
ProgramImpl::checkpoint_summary(const SequenceState& sequence, runtime::CheckpointRef checkpoint,
                                StateImageHandle state, runtime::PrefillWork rebuild_work) const {
    if (!sequence.kv) { throw std::logic_error("checkpoint summary has no KV address space"); }
    if (checkpoint.frontier == 0) {
        throw std::logic_error("checkpoint summary has an empty frontier");
    }
    if (!state_store->valid(state)) {
        throw std::logic_error("checkpoint summary has a stale StateImage");
    }
    const StateReplicaResidency state_location = state_store->residency(state);
    runtime::ReplicaResidency residency        = runtime::ReplicaResidency::DeviceOnly;
    if (state_location == StateReplicaResidency::HostOnly) {
        residency = runtime::ReplicaResidency::HostOnly;
    } else if (state_location == StateReplicaResidency::Both) {
        residency = runtime::ReplicaResidency::Both;
    } else if (state_location != StateReplicaResidency::DeviceOnly) {
        throw std::logic_error("checkpoint StateImage has no published replica");
    }
    const std::uint32_t backend_frontier =
        speculative_backend == SpeculativeBackend::Mtp      ? checkpoint.frontier - 1U
        : speculative_backend == SpeculativeBackend::DFlash ? checkpoint.frontier
                                                            : 0U;
    const std::uint32_t identity_tag = static_cast<std::uint32_t>(speculative_backend) |
                                       (static_cast<std::uint32_t>(proposal_head) << 8U) |
                                       (static_cast<std::uint32_t>(kv_storage) << 16U);
    const std::uint32_t main_pages = kv_pages_for_frontier(checkpoint.frontier);
    // The backend address space never runs windowed (reselect is gated off under speculation),
    // so its claim is always fully resident; only the text side splits.
    const auto [resident, host_only] = tiered_kv_claim(&*sequence.kv, main_pages);
    return qwen3_5::CheckpointSummary{
        .ref   = checkpoint,
        .scope = runtime::CheckpointScope::Private,
        .shortlist_key =
            {
                .digests      = sequence.prefix_digests.at(checkpoint.frontier),
                .frontier     = checkpoint.frontier,
                .identity_tag = identity_tag,
            },
        .state_residency = residency,
        .required_kv =
            {
                .main_frontier    = checkpoint.frontier,
                .backend_frontier = backend_frontier,
                .main_pages       = main_pages,
                .backend_pages    = kv_pages_for_frontier(backend_frontier),
                .resident_pages   = resident,
                .host_only_pages  = host_only,
            },
        .rebuild_work = validated_rebuild_work(rebuild_work, checkpoint.frontier),
    };
}

qwen3_5::ContinuationSummary
ProgramImpl::continuation_summary(const SequenceState& sequence) const {
    qwen3_5::ContinuationSummary summary;
    summary.long_anchors.reserve(sequence.long_anchors.size());
    populate_continuation_summary(sequence, summary);
    return summary;
}

void ProgramImpl::populate_continuation_summary(const SequenceState& sequence,
                                                qwen3_5::ContinuationSummary& summary) const {
    validate_long_anchor_ordinals(sequence.long_anchors,
                                  context_cache.max_long_anchors_per_continuation.value_or(0));
    if (summary.long_anchors.capacity() < sequence.long_anchors.size()) {
        throw std::logic_error("continuation summary backing was not reserved");
    }
    summary.endpoint.reset();
    summary.rewrite.reset();
    summary.long_anchors.clear();
    summary.active_references = 0;
    if (sequence.endpoint_valid) {
        const runtime::CheckpointRef endpoint{
            .kind     = runtime::CheckpointKind::SessionEndpoint,
            .frontier = sequence.execution_frontier,
        };
        runtime::PrefillWork endpoint_work = sequence.rebuild_work;
        summary.endpoint =
            checkpoint_summary(sequence, endpoint, sequence.state.read, endpoint_work);
    }
    if (sequence.rewrite_checkpoint.valid) {
        if (!sequence.rewrite_state) {
            throw std::logic_error("rewrite checkpoint has no StateImage");
        }
        const runtime::CheckpointRef rewrite{
            .kind     = checkpoint_kind(sequence.rewrite_checkpoint.kind),
            .frontier = sequence.rewrite_checkpoint.frontier,
        };
        summary.rewrite = checkpoint_summary(sequence, rewrite, *sequence.rewrite_state,
                                             sequence.rewrite_checkpoint.rebuild_work);
    }
    for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
        summary.long_anchors.push_back(
            checkpoint_summary(sequence,
                               runtime::CheckpointRef{.kind = runtime::CheckpointKind::LongAnchor,
                                                      .frontier = anchor.frontier,
                                                      .ordinal  = anchor.ordinal},
                               anchor.state, anchor.rebuild_work));
    }
    if (!summary.endpoint && !summary.rewrite && summary.long_anchors.empty()) {
        throw std::logic_error("private continuation has no checkpoint");
    }
    const auto* begin = continuation_states.data();
    const auto* end   = begin + continuation_capacity;
    if (&sequence >= begin && &sequence < end) {
        const std::size_t index = static_cast<std::size_t>(&sequence - begin);
        summary.active_references =
            continuation_slots[index].role == ContinuationSlotRole::Active ? 1U : 0U;
    }
}

qwen3_5::SharedPrefixSummary
ProgramImpl::shared_prefix_summary(const SharedPrefixState& shared) const {
    if (!shared.kv || !shared.identity || shared.frontier == 0 ||
        !state_store->valid(shared.state)) {
        throw std::logic_error("shared-prefix summary source is incomplete");
    }
    const StateReplicaResidency state_location = state_store->residency(shared.state);
    runtime::ReplicaResidency residency        = runtime::ReplicaResidency::DeviceOnly;
    if (state_location == StateReplicaResidency::HostOnly) {
        residency = runtime::ReplicaResidency::HostOnly;
    } else if (state_location == StateReplicaResidency::Both) {
        residency = runtime::ReplicaResidency::Both;
    } else if (state_location != StateReplicaResidency::DeviceOnly) {
        throw std::logic_error("shared-prefix StateImage has no published replica");
    }
    const std::uint32_t shared_main_pages = kv_pages_for_frontier(shared.frontier);
    const auto [shared_resident, shared_host_only] =
        tiered_kv_claim(&*shared.kv, shared_main_pages);
    return qwen3_5::SharedPrefixSummary{
        .checkpoint =
            {
                .ref =
                    {
                        .kind     = runtime::CheckpointKind::SharedStablePrefix,
                        .frontier = shared.frontier,
                    },
                .scope           = runtime::CheckpointScope::Shared,
                .shortlist_key   = shared.identity->shortlist_key,
                .state_residency = residency,
                .required_kv =
                    {
                        .main_frontier    = shared.frontier,
                        .backend_frontier = shared.backend_frontier,
                        .main_pages       = shared_main_pages,
                        .backend_pages    = kv_pages_for_frontier(shared.backend_frontier),
                        .resident_pages   = shared_resident,
                        .host_only_pages  = shared_host_only,
                    },
                .rebuild_work = validated_rebuild_work(shared.rebuild_work, shared.frontier),
            },
        .active_references = shared.active_references,
    };
}

PrefillProgress ProgramImpl::advance_prefill(SequenceHandle sequence,
                                             runtime::ExecutionTiming* failed_timing) {
    if (pending_transaction_ || !valid_sequence(sequence)) {
        throw std::logic_error("prefill sequence capability is invalid");
    }
    const std::uint32_t lane = ContractAccess::lane(sequence).value;
    if (requests[lane].lifecycle != Lifecycle::Prefilling) {
        throw std::logic_error("prefill advance requires a prefilling sequence");
    }
    try {
        runtime::PrefillStepResult step = advance_prefill_raw(lane, failed_timing);
        if (failed_timing != nullptr) { *failed_timing += step.timing; }
        return wrap_prefill(lane, std::move(step));
    } catch (...) {
        const Clock::time_point cleanup_started = Clock::now();
        clear_execution_failure_lanes(std::span<const std::uint32_t>(&lane, 1));
        if (failed_timing != nullptr) {
            failed_timing->post_host_ns += elapsed_ns(cleanup_started);
        }
        throw;
    }
}

bool ProgramImpl::can_clear_lane_strict(const SequenceState& sequence) const {
    const auto* begin = continuation_states.data();
    const auto* end   = begin + continuation_capacity;
    if (&sequence < begin || &sequence >= end || !state_store || !text_kv_addresses ||
        !text_kv_pages || !sequence.kv) {
        return false;
    }
    const std::uint32_t continuation = static_cast<std::uint32_t>(&sequence - begin);
    if (continuation_slots[continuation].role != ContinuationSlotRole::Active ||
        !text_kv_addresses->can_release_after_deactivate(sequence.kv->text) ||
        (sequence.kv->backend &&
         (!backend_kv_addresses || !backend_kv_pages ||
          !backend_kv_addresses->can_release_after_deactivate(*sequence.kv->backend)))) {
        return false;
    }

    for (std::size_t position = 0; position < sequence.shared_prefix_references.size();
         ++position) {
        const std::uint32_t index = sequence.shared_prefix_references[position];
        if (index >= shared_prefix_capacity ||
            shared_prefix_slots[index].role != SharedPrefixSlotRole::Catalogued) {
            return false;
        }
        const std::uint32_t required = static_cast<std::uint32_t>(std::count(
            sequence.shared_prefix_references.begin(),
            sequence.shared_prefix_references.begin() + static_cast<std::ptrdiff_t>(position + 1U),
            index));
        if (shared_prefix_states[index].active_references < required) { return false; }
    }

    if (!state_store->valid(sequence.state.read) || !state_store->valid(sequence.state.write) ||
        (sequence.state.fork_pending &&
         (!sequence.state.borrows_read() ||
          !state_store->can_abort_fork(sequence.state.read, sequence.state.write)))) {
        return false;
    }
    enum class ForkEndpoint : std::uint8_t { None, Source, Destination };
    const auto validate_state = [&](StateImageHandle handle, bool release_object,
                                    ForkEndpoint fork_endpoint = ForkEndpoint::None) {
        if (!state_store->valid(handle)) { return false; }
        const std::uint32_t owned = owned_checkpoint_references(sequence, handle);
        const std::uint32_t total = state_store->checkpoint_references(handle);
        if (owned > total ||
            (owned != 0 && state_store->role(handle) != StateImageRole::CheckpointImmutable &&
             fork_endpoint != ForkEndpoint::Destination)) {
            return false;
        }
        if (!release_object || total != owned) { return true; }
        if (fork_endpoint == ForkEndpoint::Source) {
            return state_store->can_release_source_after_fork_abort(sequence.state.read,
                                                                    sequence.state.write, owned);
        }
        if (fork_endpoint == ForkEndpoint::Destination) {
            return state_store->can_release_destination_after_fork_abort(
                sequence.state.read, sequence.state.write, owned);
        }
        return state_store->can_release_after_checkpoint_references(handle, owned);
    };
    const auto duplicates_binding = [&](StateImageHandle handle) {
        return handle == sequence.state.read || handle == sequence.state.write;
    };

    if (!validate_state(sequence.state.read,
                        !sequence.state.read_has_external_owner() ||
                            sequence.state.read == sequence.state.write,
                        sequence.state.fork_pending ? ForkEndpoint::Source : ForkEndpoint::None)) {
        return false;
    }
    if (sequence.state.write != sequence.state.read &&
        !validate_state(sequence.state.write, true,
                        sequence.state.fork_pending ? ForkEndpoint::Destination
                                                    : ForkEndpoint::None)) {
        return false;
    }
    if (sequence.rewrite_state && !duplicates_binding(*sequence.rewrite_state) &&
        !validate_state(*sequence.rewrite_state, true)) {
        return false;
    }
    for (std::size_t index = 0; index < sequence.long_anchors.size(); ++index) {
        const StateImageHandle handle = sequence.long_anchors[index].state;
        bool repeated                 = duplicates_binding(handle) ||
                        (sequence.rewrite_state && handle == *sequence.rewrite_state);
        for (std::size_t prior = 0; !repeated && prior < index; ++prior) {
            repeated = sequence.long_anchors[prior].state == handle;
        }
        if (!repeated && !validate_state(handle, true)) { return false; }
    }
    if (sequence.reserved_state) {
        const StateImageHandle handle = *sequence.reserved_state;
        bool repeated                 = duplicates_binding(handle) ||
                        (sequence.rewrite_state && handle == *sequence.rewrite_state);
        for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
            repeated = repeated || anchor.state == handle;
        }
        if (!repeated && !validate_state(handle, true)) { return false; }
    }
    return true;
}

void ProgramImpl::release_active_shared_references_strict(SequenceState& sequence) noexcept {
    for (const std::uint32_t index : sequence.shared_prefix_references) {
        if (index >= shared_prefix_capacity ||
            shared_prefix_slots[index].role != SharedPrefixSlotRole::Catalogued ||
            shared_prefix_states[index].active_references == 0) {
            std::terminate();
        }
        --shared_prefix_states[index].active_references;
    }
    sequence.shared_prefix_references.clear();
}

bool ProgramImpl::clear_lane_strict(SequenceState& sequence, RequestControl& request) noexcept {
    try {
        if (!can_clear_lane_strict(sequence)) { return false; }
    } catch (...) { return false; }
    const auto* begin                = continuation_states.data();
    const std::uint32_t continuation = static_cast<std::uint32_t>(&sequence - begin);
    release_active_shared_references_strict(sequence);
    release_active_sequence_kv_strict(sequence);
    release_active_sequence_state_strict(sequence);
    retire_continuation_slot(continuation);
    request.prefill.reset();
    request.lifecycle            = Lifecycle::Empty;
    request.pending              = {};
    request.active_resources     = {};
    request.optional_resources   = {};
    request.publish_continuation = true;
    return true;
}

void ProgramImpl::clear_execution_failure_lanes(std::span<const std::uint32_t> lanes) noexcept {
    // A concurrent resource transaction may pin or inspect these active owners. Engine-wide
    // cleanup aborts that transaction before releasing lanes, preserving the only safe order.
    if (has_context_transaction()) { return; }
    for (const std::uint32_t lane : lanes) {
        if (lane >= max_concurrency || active_continuations[lane] >= continuation_capacity) {
            continue;
        }
        clear_lane_best_effort(active_sequence(lane), requests[lane]);
        invalidate_lane(lane);
    }
}

void ProgramImpl::clear_lane_best_effort(SequenceState& sequence,
                                         RequestControl& request) noexcept {
    request.prefill.reset();
    request.lifecycle            = Lifecycle::Empty;
    request.pending              = {};
    request.active_resources     = {};
    request.optional_resources   = {};
    request.publish_continuation = true;
    const auto* begin            = continuation_states.data();
    const auto* end              = begin + continuation_capacity;
    if (&sequence >= begin && &sequence < end) {
        release_continuation_slot_best_effort(static_cast<std::uint32_t>(&sequence - begin));
    }
}

StateImageSelectors ProgramImpl::state_selectors(const SequenceState& sequence) const {
    if (!state_store || !state_store->valid(sequence.state.read) ||
        !state_store->valid(sequence.state.write)) {
        throw std::logic_error("sequence has no active StateImage binding");
    }
    return state_store->selectors(sequence.state.read, sequence.state.write);
}

std::uint32_t ProgramImpl::owned_checkpoint_references(const SequenceState& sequence,
                                                       StateImageHandle state) const noexcept {
    std::uint32_t references = 0;
    if (sequence.rewrite_state && *sequence.rewrite_state == state) { ++references; }
    for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
        if (anchor.state == state) { ++references; }
    }
    return references;
}

bool ProgramImpl::state_exclusive_to_sequence(const SequenceState& sequence,
                                              StateImageHandle state) const noexcept {
    if (!state_store || !state_store->valid(state)) { return false; }
    return state_store->checkpoint_references(state) ==
           owned_checkpoint_references(sequence, state);
}

void ProgramImpl::refresh_state_views(SequenceState& sequence) {
    sequence.tail_hidden               = {};
    sequence.rewrite_checkpoint_hidden = {};
    if (state_store->valid(sequence.state.read) && state_store->valid(sequence.state.write) &&
        state_store->residency(sequence.state.read) != StateReplicaResidency::HostOnly &&
        state_store->residency(sequence.state.write) != StateReplicaResidency::HostOnly) {
        const StateImageHandle committed =
            sequence.state.fork_pending ? sequence.state.read : sequence.state.write;
        sequence.tail_hidden =
            state_images->continuation_hidden_slot(state_store->physical_slot(committed));
    }
    if (sequence.rewrite_state && state_store->valid(*sequence.rewrite_state) &&
        state_store->residency(*sequence.rewrite_state) != StateReplicaResidency::HostOnly) {
        sequence.rewrite_checkpoint_hidden = state_images->continuation_hidden_slot(
            state_store->physical_slot(*sequence.rewrite_state));
    }
}

void ProgramImpl::reserve_state_entitlement(SequenceState& sequence, std::uint32_t slots) {
    const std::uint32_t owned = sequence_exclusive_state_resources(sequence).device.state_slots;
    if (slots == 0 || owned > slots) {
        throw std::logic_error("sequence StateImage entitlement is inconsistent");
    }
    if (owned == slots) { return; }
    if (slots - owned != 1 || sequence.reserved_state) {
        throw std::logic_error("sequence StateImage reservation is not a single destination");
    }
    std::optional<StateImageHandle> reserved = state_store->reserve_destination();
    if (!reserved) { throw std::bad_alloc(); }
    sequence.reserved_state = *reserved;
    if (sequence_exclusive_state_resources(sequence).device.state_slots != slots) {
        throw std::logic_error("sequence StateImage entitlement did not materialize exactly");
    }
}

void ProgramImpl::settle_state_fork(SequenceState& sequence) {
    if (!sequence.state.fork_pending) { return; }
    if (has_context_transaction()) {
        throw std::logic_error("StateImage Fork settlement overlaps a resource transaction");
    }
    const StateImageHandle source      = sequence.state.read;
    const StateImageHandle destination = sequence.state.write;
    const bool external_source         = sequence.state.read_has_external_owner();
    state_store->commit_fork(source, destination);
    sequence.state = ActiveStateBinding{.read = destination, .write = destination};
    if (!external_source && state_store->checkpoint_references(source) == 0 &&
        !state_store->release(source)) {
        throw std::logic_error("unreferenced StateImage fork source could not be released");
    }
    refresh_state_views(sequence);
}

bool ProgramImpl::has_unsettled_state_fork() const noexcept {
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        const std::uint32_t continuation = active_continuations[lane];
        if (continuation < continuation_capacity &&
            continuation_states[continuation].state.fork_pending) {
            return true;
        }
    }
    return false;
}

void ProgramImpl::release_active_sequence_state_strict(SequenceState& sequence) noexcept {
    const auto fail = []() noexcept { std::terminate(); };
    if (!state_store) { fail(); }
    try {
        if (sequence.state.fork_pending) {
            state_store->abort_fork(sequence.state.read, sequence.state.write);
        }
        if (sequence.rewrite_state) {
            state_store->release_checkpoint_reference(*sequence.rewrite_state);
        }
        for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
            state_store->release_checkpoint_reference(anchor.state);
        }

        const auto release_if_unreferenced = [&](StateImageHandle handle, bool lifetime_owned) {
            if (!lifetime_owned || !state_store->valid(handle) ||
                state_store->checkpoint_references(handle) != 0) {
                return;
            }
            if (!state_store->release(handle)) { fail(); }
        };
        const auto duplicates_binding = [&](StateImageHandle handle) {
            return handle == sequence.state.read || handle == sequence.state.write;
        };

        release_if_unreferenced(sequence.state.write, true);
        if (sequence.state.read != sequence.state.write) {
            release_if_unreferenced(sequence.state.read, !sequence.state.read_has_external_owner());
        }
        if (sequence.rewrite_state) {
            release_if_unreferenced(*sequence.rewrite_state,
                                    !duplicates_binding(*sequence.rewrite_state));
        }
        for (std::size_t index = 0; index < sequence.long_anchors.size(); ++index) {
            const StateImageHandle handle = sequence.long_anchors[index].state;
            bool repeated                 = duplicates_binding(handle) ||
                            (sequence.rewrite_state && handle == *sequence.rewrite_state);
            for (std::size_t prior = 0; !repeated && prior < index; ++prior) {
                repeated = sequence.long_anchors[prior].state == handle;
            }
            release_if_unreferenced(handle, !repeated);
        }
        if (sequence.reserved_state) {
            const StateImageHandle handle = *sequence.reserved_state;
            bool repeated                 = duplicates_binding(handle) ||
                            (sequence.rewrite_state && handle == *sequence.rewrite_state);
            for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
                repeated = repeated || anchor.state == handle;
            }
            release_if_unreferenced(handle, !repeated);
        }
    } catch (...) { fail(); }

    sequence.state          = {};
    sequence.rewrite_state  = std::nullopt;
    sequence.reserved_state = std::nullopt;
    sequence.endpoint_valid = false;
    sequence.long_anchors.clear();
    sequence.tail_hidden               = {};
    sequence.rewrite_checkpoint_hidden = {};
}

void ProgramImpl::release_sequence_state_strict(SequenceState& sequence) noexcept {
    const auto fail = []() noexcept { std::terminate(); };
    if (!state_store || sequence.state.fork_pending) { fail(); }

    try {
        if (sequence.rewrite_state) {
            state_store->release_checkpoint_reference(*sequence.rewrite_state);
        }
        for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
            state_store->release_checkpoint_reference(anchor.state);
        }

        const auto release_if_unreferenced = [&](StateImageHandle handle, bool lifetime_owned) {
            if (!lifetime_owned || !state_store->valid(handle) ||
                state_store->checkpoint_references(handle) != 0) {
                return;
            }
            if (!state_store->release(handle)) { fail(); }
        };
        const auto repeated_before_anchor = [&](std::size_t anchor_index, StateImageHandle handle) {
            if ((sequence.endpoint_valid &&
                 (handle == sequence.state.read || handle == sequence.state.write)) ||
                (sequence.rewrite_state && handle == *sequence.rewrite_state)) {
                return true;
            }
            for (std::size_t prior = 0; prior < anchor_index; ++prior) {
                if (sequence.long_anchors[prior].state == handle) { return true; }
            }
            return false;
        };

        if (sequence.endpoint_valid) {
            release_if_unreferenced(sequence.state.write, true);
            if (sequence.state.read != sequence.state.write) {
                release_if_unreferenced(sequence.state.read,
                                        !sequence.state.read_has_external_owner());
            }
        }
        if (sequence.rewrite_state) {
            const StateImageHandle handle = *sequence.rewrite_state;
            const bool duplicates_endpoint =
                sequence.endpoint_valid &&
                (handle == sequence.state.read || handle == sequence.state.write);
            release_if_unreferenced(handle, !duplicates_endpoint);
        }
        for (std::size_t index = 0; index < sequence.long_anchors.size(); ++index) {
            const StateImageHandle handle = sequence.long_anchors[index].state;
            release_if_unreferenced(handle, !repeated_before_anchor(index, handle));
        }
        if (sequence.reserved_state) {
            const StateImageHandle handle = *sequence.reserved_state;
            bool repeated                 = sequence.endpoint_valid &&
                            (handle == sequence.state.read || handle == sequence.state.write);
            repeated = repeated || (sequence.rewrite_state && handle == *sequence.rewrite_state);
            for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
                repeated = repeated || anchor.state == handle;
            }
            release_if_unreferenced(handle, !repeated);
        }
    } catch (...) { fail(); }

    sequence.state          = {};
    sequence.rewrite_state  = std::nullopt;
    sequence.reserved_state = std::nullopt;
    sequence.endpoint_valid = false;
    sequence.long_anchors.clear();
    sequence.tail_hidden               = {};
    sequence.rewrite_checkpoint_hidden = {};
}

void ProgramImpl::release_sequence_state(SequenceState& sequence) noexcept {
    if (!state_store) { return; }
    if (sequence.state.fork_pending && state_store->valid(sequence.state.read) &&
        state_store->valid(sequence.state.write)) {
        try {
            state_store->abort_fork(sequence.state.read, sequence.state.write);
        } catch (...) {}
    }

    try {
        if (sequence.rewrite_state && state_store->valid(*sequence.rewrite_state) &&
            state_store->checkpoint_references(*sequence.rewrite_state) != 0) {
            state_store->release_checkpoint_reference(*sequence.rewrite_state);
        }
        for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
            if (state_store->valid(anchor.state) &&
                state_store->checkpoint_references(anchor.state) != 0) {
                state_store->release_checkpoint_reference(anchor.state);
            }
        }
    } catch (...) {}

    const auto releasable = [&](StateImageHandle handle) { return state_store->valid(handle); };
    if (releasable(sequence.state.write)) { (void)state_store->release(sequence.state.write); }
    if (!sequence.state.read_has_external_owner() && sequence.state.read != sequence.state.write &&
        releasable(sequence.state.read)) {
        (void)state_store->release(sequence.state.read);
    }
    if (sequence.rewrite_state) {
        const StateImageHandle handle = *sequence.rewrite_state;
        const bool duplicates_binding =
            handle == sequence.state.write ||
            (!sequence.state.read_has_external_owner() && handle == sequence.state.read);
        if (!duplicates_binding && releasable(handle)) { (void)state_store->release(handle); }
    }
    for (std::size_t index = 0; index < sequence.long_anchors.size(); ++index) {
        const StateImageHandle handle = sequence.long_anchors[index].state;
        bool duplicate =
            handle == sequence.state.write ||
            (!sequence.state.read_has_external_owner() && handle == sequence.state.read) ||
            (sequence.rewrite_state && handle == *sequence.rewrite_state);
        for (std::size_t previous = 0; !duplicate && previous < index; ++previous) {
            duplicate = sequence.long_anchors[previous].state == handle;
        }
        if (!duplicate && releasable(handle)) { (void)state_store->release(handle); }
    }
    if (sequence.reserved_state) {
        const StateImageHandle handle = *sequence.reserved_state;
        bool duplicate =
            handle == sequence.state.write ||
            (!sequence.state.read_has_external_owner() && handle == sequence.state.read) ||
            (sequence.rewrite_state && handle == *sequence.rewrite_state);
        for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
            duplicate = duplicate || anchor.state == handle;
        }
        if (!duplicate && releasable(handle)) { (void)state_store->release(handle); }
    }
    sequence.state          = {};
    sequence.rewrite_state  = std::nullopt;
    sequence.reserved_state = std::nullopt;
    sequence.endpoint_valid = false;
    sequence.long_anchors.clear();
    sequence.tail_hidden               = {};
    sequence.rewrite_checkpoint_hidden = {};
}

void ProgramImpl::release_active_shared_references(SequenceState& sequence) noexcept {
    for (const std::uint32_t index : sequence.shared_prefix_references) {
        if (index >= shared_prefix_capacity ||
            shared_prefix_slots[index].role != SharedPrefixSlotRole::Catalogued ||
            shared_prefix_states[index].active_references == 0) {
            continue;
        }
        --shared_prefix_states[index].active_references;
    }
    sequence.shared_prefix_references.clear();
}

qwen3_5::PagedKVCache* ProgramImpl::backend_kv_cache() noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) { return decoder->mtp_cache(); }
    if (dflash && dflash->full) { return &*dflash->full; }
    return nullptr;
}

const qwen3_5::PagedKVCache* ProgramImpl::backend_kv_cache() const noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) { return decoder->mtp_cache(); }
    if (dflash && dflash->full) { return &*dflash->full; }
    return nullptr;
}

std::uint32_t ProgramImpl::backend_kv_valid(const SequenceState& sequence) const noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) { return sequence.mtp_kv_valid; }
    if (speculative_backend == SpeculativeBackend::DFlash) {
        return sequence.dflash_context_frontier;
    }
    return 0;
}

void ProgramImpl::resize_sequence_kv_entitlement(SequenceState& sequence, std::uint32_t text_pages,
                                                 std::uint32_t backend_pages) {
    if (!sequence.kv || text_pages == 0 ||
        (sequence.kv->backend.has_value() != (backend_pages != 0))) {
        throw std::invalid_argument("KV resize entitlement does not match the sequence bundle");
    }
    text_kv_addresses->resize_entitlement(sequence.kv->text, text_pages);
    if (sequence.kv->backend) {
        backend_kv_addresses->resize_entitlement(*sequence.kv->backend, backend_pages);
    }
}

void ProgramImpl::bind_sequence_kv(SequenceState& sequence) {
    if (!sequence.kv) { throw std::logic_error("KV allocation bundle is unavailable"); }
    const std::int32_t row = static_cast<std::int32_t>(sequence.lane);
    const bool text_active = text_kv_addresses->active(sequence.kv->text);
    const bool backend_active =
        sequence.kv->backend && backend_kv_addresses->active(*sequence.kv->backend);
    if (sequence.kv->backend && text_active != backend_active) {
        throw std::logic_error("KV address-space activation is not bundle-atomic");
    }
    try {
        if (!text_active) {
            text_kv_addresses->activate(sequence.kv->text,
                                        text_kv_addresses->mapped_pages(sequence.kv->text), row,
                                        device.stream);
            if (sequence.kv->backend) {
                backend_kv_addresses->activate(
                    *sequence.kv->backend,
                    backend_kv_addresses->mapped_pages(*sequence.kv->backend), row,
                    device.stream);
            }
        }
        set_device_i32(io.text_kv_table_row, text_kv_addresses->bound_row(sequence.kv->text));
        set_device_i32(io.backend_kv_table_row,
                       sequence.kv->backend ? backend_kv_addresses->bound_row(*sequence.kv->backend)
                                            : 0);
    } catch (...) {
        if (!text_active) {
            if (sequence.kv->backend && backend_kv_addresses->active(*sequence.kv->backend)) {
                backend_kv_addresses->deactivate(*sequence.kv->backend);
            }
            if (text_kv_addresses->active(sequence.kv->text)) {
                text_kv_addresses->deactivate(sequence.kv->text);
            }
        }
        throw;
    }
}

void ProgramImpl::unbind_sequence_kv(SequenceState& sequence) noexcept {
    if (!sequence.kv) { return; }
    try {
        if (sequence.kv->backend && backend_kv_addresses->active(*sequence.kv->backend)) {
            backend_kv_addresses->deactivate(*sequence.kv->backend);
        }
    } catch (...) {}
    try {
        if (text_kv_addresses->active(sequence.kv->text)) {
            text_kv_addresses->deactivate(sequence.kv->text);
        }
    } catch (...) {}
}

void ProgramImpl::ensure_sequence_kv_mapped(SequenceState& sequence, std::uint32_t main_tokens,
                                            std::uint32_t backend_tokens) {
    if (!sequence.kv || main_tokens > capacity || backend_tokens > capacity) {
        throw std::logic_error("KV materialization request is outside the sequence bundle");
    }
    if (backend_tokens != 0 && !sequence.kv->backend) {
        throw std::logic_error("backend KV materialization requested without an allocation");
    }
    text_kv_addresses->ensure_mapped_to_tokens(sequence.kv->text, main_tokens, device.stream);
    if (backend_tokens != 0) {
        backend_kv_addresses->ensure_mapped_to_tokens(*sequence.kv->backend, backend_tokens,
                                                      device.stream);
    }
}

void ProgramImpl::commit_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                                     std::uint32_t backend_tokens) {
    if (!sequence.kv || main_tokens > capacity || backend_tokens > capacity ||
        (backend_tokens != 0 && !sequence.kv->backend)) {
        throw std::logic_error("KV commit request is outside the sequence bundle");
    }
    text_kv_addresses->commit_frontier(sequence.kv->text, main_tokens);
    if (sequence.kv->backend) {
        backend_kv_addresses->commit_frontier(*sequence.kv->backend, backend_tokens);
    }
}

void ProgramImpl::prepare_kvmem_capture() {
    if (!retrieval_selection_active()) { return; }
    if (kvmem_sums_device) { return; }
    const std::optional<AttentionConfig>& attention = parameters.model.config().text.attention;
    if (!attention) { throw std::logic_error("Qwen3.5 text model has no attention geometry"); }
    if (kvmem_config.block_tokens == 0) {
        throw std::logic_error("kvmem capture needs a positive block size");
    }
    const std::uint32_t key_width = static_cast<std::uint32_t>(attention->key_width());
    // One staging column is either a decode lane (each contributes exactly one block) or the widest
    // prefill sub-chunk's block span. Segment capacity is one per prefill sub-chunk, which the chunk
    // loop bounds; a chunk is consumed by at most ceil(tokens / block) + 1 reductions, and one extra
    // slot keeps a bound violation a throw at the capture site rather than a silent overwrite.
    const std::uint32_t decode_blocks = ninfer::kMaximumConcurrency;
    const std::uint32_t prefill_blocks =
        (prefill_chunk / kvmem_config.block_tokens) + 1U;
    // MTP verify stages one column per verify column (up to W x B) on top of the per-lane column
    // of ordinary decode; the extra staging stays allocated but unused for other backends.
    const std::uint32_t verify_blocks =
        speculative_backend == SpeculativeBackend::Mtp
            ? static_cast<std::uint32_t>(ninfer::kMaximumConcurrency) * kMtpVerifyMaximumWidth
            : 0U;
    const std::uint32_t capacity = std::max({decode_blocks, prefill_blocks, verify_blocks});
    kvmem_capture_key_width       = key_width;
    kvmem_capture_capacity_blocks = capacity;
    kvmem_capture_segment_capacity = prefill_blocks + 1U;
    kvmem_capture_segments.resize(kvmem_capture_segment_capacity);
    const std::size_t bytes =
        sizeof(float) * static_cast<std::size_t>(key_width) * static_cast<std::size_t>(capacity);
    kvmem_sums_host.emplace(bytes);
    kvmem_sums_device.emplace(bytes);
}

execution::KVMemCaptureTarget* ProgramImpl::kvmem_capture_target(execution::KVMemCaptureMode mode,
                                                                std::uint32_t lanes) {
    if (mode == execution::KVMemCaptureMode::None || !retrieval_selection_active()) { return nullptr; }
    if (!kvmem_sums_device || !kvmem_sums_host || kvmem_capture_key_width == 0) {
        throw std::logic_error("kvmem capture target used before its staging was prepared");
    }
    if (mode == execution::KVMemCaptureMode::SequenceBatch && lanes > kvmem_capture_capacity_blocks) {
        throw std::logic_error("kvmem capture staging cannot hold this decode batch");
    }
    execution::KVMemCaptureTarget& target = kvmem_capture_binding;
    target.sums              = static_cast<float*>(kvmem_sums_device->p);
    target.host              = static_cast<float*>(kvmem_sums_host->data());
    target.segments          = kvmem_capture_segments.data();
    target.segment_capacity  = kvmem_capture_segment_capacity;
    target.capacity_blocks   = kvmem_capture_capacity_blocks;
    target.key_width         = kvmem_capture_key_width;
    target.block_tokens      = kvmem_config.block_tokens;
    target.mode              = mode;
    // run_layers numbers only Full Attention blocks, so compact index 0 is the first one.
    target.capture_layer     = 0;
    return &target;
}

void ProgramImpl::accumulate_kvmem_scores(SequenceState& sequence,
                                          std::span<const execution::KVMemCaptureSegment> segments) {
    if (!retrieval_selection_active() || segments.empty()) { return; }
    const std::optional<AttentionConfig>& attention = parameters.model.config().text.attention;
    if (!attention) { throw std::logic_error("Qwen3.5 text model has no attention geometry"); }
    const std::uint32_t key_width    = kvmem_capture_key_width;
    const std::uint32_t block_tokens = kvmem_config.block_tokens;
    if (!sequence.kv) { throw std::logic_error("kvmem mean-K accumulation without a text KV bundle"); }
    if (!sequence.kvmem) {
        // The block mirror is what bounds which blocks exist; without it a capture range could name
        // a block the store never registered. Every capture route reselects first, so this is a
        // guard on the invariant rather than an expected state.
        throw std::logic_error("kvmem mean-K accumulation before the block mirror exists");
    }
    if (!sequence.kvmem_scores) {
        sequence.kvmem_scores.emplace(block_tokens, key_width, attention->head_dim);
    }
    qwen3_5::detail::KVMemScoreStore& scores = *sequence.kvmem_scores;
    const float* host                        = static_cast<const float*>(kvmem_sums_host->data());

    // Fold one staged block into the store and promote it once its token count is complete. A block
    // the block mirror does not know yet, or that already has a mean, contributes nothing new.
    const auto fold = [&](std::uint32_t block_id, std::uint32_t offset, std::uint32_t count) {
        if (count == 0 || block_id >= sequence.kvmem->block_count()) { return; }
        if (scores.has_mean(block_id)) { return; }
        scores.accumulate(block_id, host + offset, count);
        if (scores.captured_tokens(block_id) == block_tokens) { scores.finalize(block_id); }
    };

    for (const execution::KVMemCaptureSegment& segment : segments) {
        if (segment.tokens == 0) { continue; }
        // kvmem_block_shares derives the same block partition the reduction used, so column b of the
        // segment lands on the block share b covers.
        const std::vector<qwen3_5::detail::KVMemBlockShare> shares = qwen3_5::detail::kvmem_block_shares(
            segment.first_position, static_cast<std::int64_t>(segment.tokens), block_tokens);
        for (std::size_t b = 0; b < shares.size(); ++b) {
            fold(shares[b].block_id,
                 segment.offset + static_cast<std::uint32_t>(b) * key_width, shares[b].tokens);
        }
    }
}

void ProgramImpl::accumulate_kvmem_decode_scores(SequenceState& sequence,
                                                 std::uint32_t column,
                                                 std::uint32_t token_position) {
    if (!retrieval_selection_active()) { return; }
    const std::optional<AttentionConfig>& attention = parameters.model.config().text.attention;
    if (!attention) { throw std::logic_error("Qwen3.5 text model has no attention geometry"); }
    const std::uint32_t key_width    = kvmem_capture_key_width;
    const std::uint32_t block_tokens = kvmem_config.block_tokens;
    if (!sequence.kv) { throw std::logic_error("kvmem mean-K accumulation without a text KV bundle"); }
    if (!sequence.kvmem) {
        throw std::logic_error("kvmem mean-K accumulation before the block mirror exists");
    }
    if (!sequence.kvmem_scores) {
        sequence.kvmem_scores.emplace(block_tokens, key_width, attention->head_dim);
    }
    qwen3_5::detail::KVMemScoreStore& scores = *sequence.kvmem_scores;
    // `column` indexes the staging column holding this token's key-sum row (one key_width-float
    // column per captured token under MTP verify; one per lane under ordinary decode), so the
    // host offset is column * key_width. `token_position` is the absolute token index whose block
    // receives this token's contribution.
    const float* host = static_cast<const float*>(kvmem_sums_host->data()) +
                        static_cast<std::size_t>(column) * key_width;
    const std::uint32_t block_id = token_position / block_tokens;
    // The lane contributed exactly one token, so the mean completes only when this token fills the
    // block; a block already promoted (or not yet mirrored) is skipped.
    if (block_id >= sequence.kvmem->block_count() || scores.has_mean(block_id)) { return; }
    const std::uint32_t captured = scores.captured_tokens(block_id);
    if (captured >= block_tokens) { return; }
    scores.accumulate(block_id, host, 1);
    if (scores.captured_tokens(block_id) == block_tokens) { scores.finalize(block_id); }
}

KVMemPlan ProgramImpl::select_window_blocks(SequenceState& sequence, KVMemBlockStore& store) {
    if (!retrieval_selection_active()) { return store.select_recency(); }
    const std::optional<AttentionConfig>& attention = parameters.model.config().text.attention;
    if (!attention) { throw std::logic_error("Qwen3.5 text model has no attention geometry"); }
    const auto key_width = static_cast<std::uint32_t>(attention->key_width());
    if (!sequence.kvmem_scores) {
        sequence.kvmem_scores.emplace(kvmem_config.block_tokens, key_width, attention->head_dim);
    }
    KVMemScoreStore& scores = *sequence.kvmem_scores;
    // The query is the newest completed block: the agent step's own text is the most recent
    // context, which is exactly what a retrieval round wants to find more of. Until one exists
    // (the first chunk of a request) there is nothing to rank against and recency is exact.
    std::uint32_t query_block = scores.block_count();
    while (query_block > 0 && !scores.has_mean(query_block - 1U)) { --query_block; }
    if (query_block == 0) { return store.select_recency(); }
    const std::vector<float>& query = scores.mean(query_block - 1U);
    store.set_scores(scores.score(query.data(), key_width));
    return store.select_topk();
}

std::int32_t ProgramImpl::reselect_sequence_window(SequenceState& sequence, std::uint32_t call_min,
                                                   std::uint32_t call_max) {
    // Masked draft backends read this table with absolute positions and have no cache-position
    // pipeline yet, so they keep the identity window (stage 6b). MTP verification consumes the
    // window row shift through the verify/AR cache positions instead (mtp-kvmem-plan.md D4).
    if (is_masked_draft_backend(speculative_backend)) { return 0; }
    if (!sequence.kv) { throw std::logic_error("kvmem reselect without a text KV bundle"); }
    if (call_min > call_max || call_max == std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("kvmem reselect range is invalid");
    }
    if (!sequence.kvmem) { sequence.kvmem.emplace(kvmem_config); }
    KVMemBlockStore& store = *sequence.kvmem;
    const std::uint32_t total = call_max + 1;
    if (store.total_tokens() < total) {
        store.register_append(total - store.total_tokens());
    } else if (store.total_tokens() > total) {
        store.truncate_to(total);
    }
    // A truncated frontier invalidates the blocks past it, means included: their token positions
    // no longer exist, and a later append re-captures them from the model.
    if (sequence.kvmem_scores && sequence.kvmem_scores->block_count() > store.block_count()) {
        sequence.kvmem_scores->truncate(store.block_count());
    }
    const KVMemPlan plan   = select_window_blocks(sequence, store);
    const KVMemWindow window = store.window_for(plan, call_min, call_max);
    store.apply(plan);
    // Tier publication: restore re-entries, republish (window-aware), demote the dropped pages.
    // window_for proved the call sits on one side of the window, where rel() is either the
    // identity (below sink_end) or one constant page shift (at or above recent_begin).
    apply_window_transition(sequence, window);
    return static_cast<std::int32_t>(window.rel(call_min)) -
           static_cast<std::int32_t>(call_min);
}

bool ProgramImpl::stage_in_demoted_pages(const KVWindowSpace& space, KVAddressSpaceHandle handle,
                                         std::span<const std::uint32_t> logical_pages) {
    if (logical_pages.empty()) { return false; }
    if (!host_kv_extents) { throw std::logic_error("KV stores are unavailable for stage-in"); }
    KVAddressSpaceStore&                   addresses = space.addresses;
    LogicalKVPageStore&                    pages     = space.pages;
    bool enqueued = false;
    for (const std::uint32_t page : logical_pages) {
        const LogicalKVPageHandle logical = addresses.logical_page(handle, page);
        // Holes left Device-resident by an abort heal (or a window change whose stage-out has
        // not run yet) need no transfer; publish just never saw them demoted-without-device.
        if (pages.device_resident(logical)) { continue; }
        // Read the Host replica before reserving so a stale page fails before any pool claim
        // moves; reserve_stage_in re-checks currency against the live page.
        const HostKVPageReplica replica = pages.host_replica(logical);
        const DeviceKVPageHandle destination = addresses.reserve_stage_in(handle, page);
        try {
            const HostKVAllocationConstView source =
                host_kv_extents->view(replica.extent).subview(replica.page_offset, 1U);
            const std::array<DeviceKVPageHandle, 1> destinations{destination};
            pages.physical_pool().copy_from_host(
                source, std::span<const DeviceKVPageHandle>(destinations), device.transfer_stream);
            pages.publish_device_replica(logical);
        } catch (...) {
            // Land any copy that was enqueued before the failure so abort_stage_in never returns
            // the destination while the transfer stream still writes into it.
            (void)cudaStreamSynchronize(device.transfer_stream);
            addresses.abort_stage_in(handle, page);
            throw;
        }
        enqueued = true;
    }
    return enqueued;
}

void ProgramImpl::stage_out_window_holes(const KVWindowSpace& space, KVAddressSpaceHandle handle,
                                         const KVMemWindow& window) {
    if (!host_kv_extents) { throw std::logic_error("KV stores are unavailable for stage-out"); }
    KVAddressSpaceStore& addresses = space.addresses;
    LogicalKVPageStore&  pages     = space.pages;
    const std::uint32_t page_size = static_cast<std::uint32_t>(kPagedKVPageSize);
    const std::uint32_t mapped    = addresses.mapped_pages(handle);
    const auto retained           = [&](std::uint32_t page) {
        return page * page_size < window.sink_end || page * page_size >= window.recent_begin;
    };
    // Pages outside the window that still hold Device residency: drop directly when a current
    // Host replica already stands behind them, back up first otherwise. A stale Host replica
    // would mean hole content changed, which the store forbids; leaving such a page Device
    // resident only overshoots the budget, so it is skipped rather than risked.
    std::vector<std::uint32_t> ready;
    std::vector<std::vector<std::uint32_t>> backup_runs;
    for (std::uint32_t page = 0; page < mapped; ++page) {
        if (retained(page) || !addresses.demoted_page(handle, page)) { continue; }
        const LogicalKVPageHandle logical = addresses.logical_page(handle, page);
        if (!pages.device_resident(logical)) { continue; }
        if (pages.host_resident(logical)) {
            if (pages.host_replica_current(logical)) { ready.push_back(page); }
            continue;
        }
        if (!backup_runs.empty() && backup_runs.back().back() + 1U == page) {
            backup_runs.back().push_back(page);
        } else {
            backup_runs.push_back({page});
        }
    }
    if (ready.empty() && backup_runs.empty()) { return; }

    struct PendingBackup {
        HostKVExtentReservation reservation;
        std::vector<DeviceKVPageHandle> sources;
        std::vector<LogicalKVPageHandle> pages;
    };
    std::vector<PendingBackup> backups;
    backups.reserve(backup_runs.size());
    try {
        for (const std::vector<std::uint32_t>& run : backup_runs) {
            std::vector<LogicalKVPageHandle> run_pages;
            run_pages.reserve(run.size());
            for (const std::uint32_t page : run) {
                run_pages.push_back(addresses.logical_page(handle, page));
            }
            std::optional<HostKVExtentReservation> reserved =
                host_kv_extents->prepare(pages, run_pages);
            if (!reserved) { throw std::bad_alloc(); }
            // Move the reservation into backups before enqueueing its copy: the catch below can
            // only free extents after the transfer stream has drained, and only backups (an
            // outer-scope object) outlives the propagation into that catch.
            backups.push_back({std::move(*reserved), {}, std::move(run_pages)});
            PendingBackup& entry = backups.back();
            // The sizing overload sizes `sources` from the reservation itself. The two-argument form
            // only fills a caller-sized span, so handing it the still-empty vector that
            // push_back just wrote would reject every window slide.
            entry.sources = host_kv_extents->device_sources(entry.reservation);
            pages.physical_pool().copy_to_host(
                entry.sources, host_kv_extents->writable_view(entry.reservation),
                device.transfer_stream);
        }
    } catch (...) {
        // Drain already-enqueued D2H copies before backups unwinds and returns its Host extents
        // to the arena for reuse.
        (void)cudaStreamSynchronize(device.transfer_stream);
        throw;
    }
    // Host data must land before a Host replica claims currency, and the Device slots must not
    // be recycled while a copy still reads them. Exits are already gone from the execution
    // table (set_window published before this runs), so no compute-stream reader is involved.
    (void)cudaStreamSynchronize(device.transfer_stream);
    for (PendingBackup& backup : backups) {
        (void)host_kv_extents->publish(std::move(backup.reservation));
    }
    const auto drop = [&](std::uint32_t page) {
        // can_drop gates the shared/pinned cases; a skip keeps the page Device-resident, which
        // is a bounded budget overshoot, never a correctness problem.
        (void)pages.drop_device_replica(addresses.logical_page(handle, page));
    };
    for (const std::uint32_t page : ready) { drop(page); }
    for (const std::vector<std::uint32_t>& run : backup_runs) {
        for (const std::uint32_t page : run) { drop(page); }
    }
}

void ProgramImpl::stage_in_all_holes(const KVWindowSpace& space, KVAddressSpaceHandle handle) {
    const std::uint32_t mapped = space.addresses.mapped_pages(handle);
    std::vector<std::uint32_t> holes;
    for (std::uint32_t page = 0; page < mapped; ++page) {
        if (space.addresses.demoted_page(handle, page)) { holes.push_back(page); }
    }
    if (stage_in_demoted_pages(space, handle, holes)) {
        (void)cudaStreamSynchronize(device.transfer_stream);
    }
}

void ProgramImpl::apply_window_space_transition(const KVWindowSpace& space, KVAddressSpaceHandle handle,
                                                const KVMemWindow& window) {
    KVAddressSpaceStore& addresses = space.addresses;
    LogicalKVPageStore&  pages     = space.pages;
    const std::uint32_t page_size = static_cast<std::uint32_t>(kPagedKVPageSize);
    const std::uint32_t mapped    = addresses.mapped_pages(handle);
    const auto retained           = [&](std::uint32_t page) {
        return page * page_size < window.sink_end || page * page_size >= window.recent_begin;
    };
    std::vector<std::uint32_t> entries;
    for (std::uint32_t page = 0; page < mapped; ++page) {
        if (retained(page) && addresses.demoted_page(handle, page) &&
            !pages.device_resident(addresses.logical_page(handle, page))) {
            entries.push_back(page);
        }
    }
    const bool restored = stage_in_demoted_pages(space, handle, entries);
    // A moved window is the only reason to pay for a republish; the address keeps its window
    // across deactivate and activation republishes window-aware through commit_activation.
    const KVMemWindow current = addresses.window(handle);
    if (current.sink_end != window.sink_end || current.recent_begin != window.recent_begin ||
        current.window_tokens != window.window_tokens) {
        addresses.set_window(handle, window, device.stream);
    }
    stage_out_window_holes(space, handle, window);
    // Stage-in copies are read by attention through the freshly published table before reselect
    // returns; the stage-out sync above does not order them when no backup ran.
    if (restored) { (void)cudaStreamSynchronize(device.transfer_stream); }
}

void ProgramImpl::apply_window_transition(SequenceState& sequence, const KVMemWindow& window) {
    if (!sequence.kv) { throw std::logic_error("kvmem window transition without a text KV bundle"); }
    if (!text_kv_addresses || !text_kv_pages) {
        throw std::logic_error("text KV stores are unavailable for a window transition");
    }
    // Phase 6a: both address spaces follow the same window under MTP staging; the draft pool
    // reads its own table with the same row shift as the text pool (D1). Backend space exists
    // exactly when MTP staging is active.
    apply_window_space_transition(KVWindowSpace{*text_kv_addresses, *text_kv_pages},
                                  sequence.kv->text, window);
    if (sequence.kv->backend) {
        if (!backend_kv_addresses || !backend_kv_pages) {
            throw std::logic_error("backend KV stores are unavailable for a window transition");
        }
        // Prefill reselects before the first commit, when the draft pool holds no tokens and the
        // identity table ensure_materialize published is still the right answer: verify cannot
        // read a row below its own committed frontier, so a window over positions this space has
        // never committed carries no information and would fail set_window's coverage check
        // (window_tokens against zero mapped pages). The commit reselect publishes the real
        // window in the same round the pool first gains rows.
        if (backend_kv_addresses->committed_frontier(*sequence.kv->backend) != 0) {
            apply_window_space_transition(KVWindowSpace{*backend_kv_addresses, *backend_kv_pages},
                                          *sequence.kv->backend, window);
        }
    }
}

void ProgramImpl::trim_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                                   std::uint32_t backend_tokens) {
    if (!sequence.kv || main_tokens > capacity || backend_tokens > main_tokens) {
        throw std::logic_error("KV trim request is outside the sequence bundle");
    }
    if (backend_tokens != 0 && !sequence.kv->backend) {
        throw std::logic_error("backend KV trim requested without an allocation");
    }
    text_kv_addresses->destructive_truncate(sequence.kv->text, main_tokens);
    if (sequence.kv->backend) {
        backend_kv_addresses->destructive_truncate(*sequence.kv->backend, backend_tokens);
    }
}

void ProgramImpl::release_sequence_growth_entitlement(SequenceState& sequence) noexcept {
    if (!sequence.kv) { return; }
    try {
        text_kv_addresses->release_growth_entitlement(sequence.kv->text);
        if (sequence.kv->backend) {
            backend_kv_addresses->release_growth_entitlement(*sequence.kv->backend);
        }
    } catch (...) {}
}

void ProgramImpl::release_active_sequence_kv_strict(SequenceState& sequence) noexcept {
    if (!sequence.kv || !text_kv_addresses ||
        !text_kv_addresses->can_release_after_deactivate(sequence.kv->text) ||
        (sequence.kv->backend &&
         (!backend_kv_addresses ||
          !backend_kv_addresses->can_release_after_deactivate(*sequence.kv->backend)))) {
        std::terminate();
    }
    if (sequence.kv->backend &&
        !backend_kv_addresses->release_after_deactivate(*sequence.kv->backend)) {
        std::terminate();
    }
    if (!text_kv_addresses->release_after_deactivate(sequence.kv->text)) { std::terminate(); }
    sequence.kv.reset();
    if (host_kv_extents) { (void)host_kv_extents->release_unreferenced(); }
}

void ProgramImpl::release_sequence_kv_strict(SequenceState& sequence) noexcept {
    if (!sequence.kv || !text_kv_addresses || !text_kv_addresses->can_release(sequence.kv->text)) {
        std::terminate();
    }
    if (sequence.kv->backend &&
        (!backend_kv_addresses || !backend_kv_addresses->can_release(*sequence.kv->backend))) {
        std::terminate();
    }
    if (sequence.kv->backend && !backend_kv_addresses->release(*sequence.kv->backend)) {
        std::terminate();
    }
    if (!text_kv_addresses->release(sequence.kv->text)) { std::terminate(); }
    sequence.kv.reset();
    if (host_kv_extents) { (void)host_kv_extents->release_unreferenced(); }
}

void ProgramImpl::release_sequence_kv(SequenceState& sequence) noexcept {
    if (!sequence.kv) { return; }
    unbind_sequence_kv(sequence);
    if (sequence.kv->backend && backend_kv_addresses) {
        (void)backend_kv_addresses->release(*sequence.kv->backend);
    }
    if (text_kv_addresses) { (void)text_kv_addresses->release(sequence.kv->text); }
    sequence.kv.reset();
    if (host_kv_extents) { (void)host_kv_extents->release_unreferenced(); }
}

qwen3_5::PagedKVCacheView ProgramImpl::text_kv_view(const SequenceState& sequence) const {
    if (!sequence.kv || !text_kv_addresses->active(sequence.kv->text)) {
        throw std::logic_error("sequence has no active KV execution mapping");
    }
    return decoder->text_kv.execution_view(text_kv_addresses->execution_row(sequence.kv->text));
}

qwen3_5::PagedKVCacheView ProgramImpl::mtp_kv_view(const SequenceState& sequence) const {
    if (speculative_backend != SpeculativeBackend::Mtp) { return {}; }
    if (decoder->mtp_cache() == nullptr || !sequence.kv || !sequence.kv->backend ||
        !backend_kv_addresses->active(*sequence.kv->backend)) {
        throw std::logic_error("sequence has no active MTP KV execution mapping");
    }
    return decoder->mtp_cache()->execution_view(
        backend_kv_addresses->execution_row(*sequence.kv->backend));
}

void ProgramImpl::set_device_i32(Tensor& tensor, std::int32_t value) {
    CUDA_CHECK(
        cudaMemcpyAsync(tensor.data, &value, sizeof(value), cudaMemcpyHostToDevice, device.stream));
}

void ProgramImpl::ordered_reset(SequenceState& sequence) {
    if (!state_store->valid(sequence.state.write)) {
        throw std::logic_error("pre-reset StateImage reservation is missing");
    } else {
        if (sequence.state.fork_pending || sequence.state.read != sequence.state.write ||
            state_store->role(sequence.state.write) != StateImageRole::ActiveMutable) {
            throw std::logic_error("StateImage reset requires a private mutable destination");
        }
    }
    refresh_state_views(sequence);
    work.reset();
    set_device_i32(io.pos, 0);
    set_device_i32(io.rope_pos, 0);
    set_device_i32(io.rope_delta, 0);
    if (io.mtp) { set_device_i32(io.mtp->position, 0); }
    sequence.text_kv_valid           = 0;
    sequence.mtp_kv_valid            = 0;
    sequence.dflash_context_frontier = 0;
}


} // namespace ninfer::models::qwen3_5::detail
