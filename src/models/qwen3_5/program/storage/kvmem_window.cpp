#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context.h"
#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/storage/kvmem_transfer_trace.h"
#include "models/qwen3_5/program/storage/checkpoint_recompute_policy.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <unordered_set>
#include <unordered_map>

namespace ninfer::models::qwen3_5::detail {
namespace {
constexpr std::uint32_t page_tokens = 64;
constexpr std::size_t transfer_staging_limit = 64ULL << 20;
std::atomic<std::uint64_t> next_window_session{1};

std::uint32_t valid_columns(std::uint32_t logical_page, std::uint32_t frontier) {
    return std::min(page_tokens, frontier - logical_page * page_tokens);
}

void publish_memory_checkpoint_source(KvmemWindowState& window,
                                     std::shared_ptr<KvmemPrefixCheckpoint> checkpoint,
                                     bool query_source) {
    // Create both entries before borrowing their vector-backed references. A
    // forked catalog's capacity can be smaller than its configured slot limit.
    (void)window.checkpoints.slot(kvmem::CheckpointPurpose::QueryReplay);
    (void)window.checkpoints.slot(kvmem::CheckpointPurpose::PromptTail);
    publish_checkpoint_execution_source(
        window.checkpoints.slot(kvmem::CheckpointPurpose::QueryReplay).data,
        window.checkpoints.slot(kvmem::CheckpointPurpose::PromptTail).data,
        std::move(checkpoint), query_source, window.query_replay_active);
}
} // namespace

// This bridge borrows Program's existing native pool, address space, stream and
// reservation. It owns no device allocator. Its lifetime is one safe boundary;
// durable history and versions travel with SequenceState.
class WindowMemoryBackend final : public kvmem::MemoryBackend {
public:
    WindowMemoryBackend(ProgramImpl& program, SequenceState& sequence, std::uint32_t end, bool resident_native = false)
        : program(program), sequence(sequence), end(end), resident_native(resident_native),
          layout(plan_host_kv_page_layout(program.text_kv_pages->physical_pool().geometry())) {
        if (program.backend_kv_pages) {
            if (sequence.mtp_kv_valid != sequence.text_kv_valid) {
                throw std::logic_error("KVMem dual-pool transfer requires a committed safe boundary");
            }
            mtp_layout = plan_host_kv_page_layout(program.backend_kv_pages->physical_pool().geometry());
        }
        page_bytes = layout.page_stride + (mtp_layout ? mtp_layout->page_stride : 0);
        auto& stamp = sequence.window.stamp;
        if (!stamp.session.id) {
            stamp = {true, {next_window_session.fetch_add(1), 1}, 1, 0, 0, 1};
        }
        if (!stamp.valid) { throw std::logic_error("KVMem sequence requires fresh prefill"); }
        if (stamp.frontier != sequence.text_kv_valid) {
            stamp.frontier = sequence.text_kv_valid;
            ++stamp.history_revision;
        }
        const auto count = kv_pages_for_tokens(sequence.text_kv_valid);
        // Append preparation may already own buffers for the next GPU chunk.
        // Keep them until publication or explicit history truncation.
        if (sequence.window.archive.size() < count) sequence.window.archive.resize(count);
        if (sequence.window.host_versions.size() < count) sequence.window.host_versions.resize(count);
    }

    kvmem::MemoryDescriptor descriptor() const override {
        kvmem::MemoryDescriptor d;
        std::string model_id;
        constexpr char hex[] = "0123456789abcdef";
        for (const auto value : program.parameters.model.info().artifact_id) {
            const auto byte = std::to_integer<unsigned char>(value);
            model_id += hex[byte >> 4]; model_id += hex[byte & 15];
        }
        d.identity = {"ninfer", program.backend_kv_pages ? "kvmem-main-mtp-pages-v3" : "kvmem-native-pages-v2",
            model_id, "qwen3.5-kvmem-state-image-v4"};
        // StateImage contains every GDN shard, including ranks without KV heads.
        // Bind checkpoints to this partition rather than only to its byte size.
        for (std::size_t i = 0; i < program.state_images->shard_count(); ++i) {
            const auto& shard = program.state_images->shard(i);
            d.identity.state_schema += ":" + std::to_string(shard.rank) + "," +
                std::to_string(shard.first_layer) + "," + std::to_string(shard.layers) + "," +
                std::to_string(shard.part) + "," + std::to_string(shard.value_heads) + "," +
                std::to_string(shard.conv_channels);
        }
        const char* layout_name = nullptr;
        switch (program.kv_storage) {
        case KvCacheStorage::BFloat16: layout_name = "ninfer.bf16.page.v1"; break;
        case KvCacheStorage::Int8Group64: layout_name = "ninfer.int8-g64.page.v1"; break;
        case KvCacheStorage::Fp8E4M3Row256: layout_name = "ninfer.fp8-e4m3-row256.page.v1"; break;
        case KvCacheStorage::Nvfp4Group16: layout_name = "ninfer.nvfp4-g16.page.v1"; break;
        case KvCacheStorage::Fp8KeyNvfp4Value: layout_name = "ninfer.k8v4-fp8-nvfp4.page.v1"; break;
        case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64:
            layout_name = "ninfer.rk8v4-k64-v32.page.v1";
            break;
        case KvCacheStorage::RotatedLloyd4KeyInt4Value:
            layout_name = "ninfer.rk4v4-lloyd-k64-v32.page.v1";
            break;
        case KvCacheStorage::RotatedInt4KeyInt4ValueE8:
            layout_name = "ninfer.rk4v4-e8-k64-v32.page.v1";
            break;
        case KvCacheStorage::RotatedE8RootKeyInt4Value:
            layout_name = "ninfer.rk2v4-e8-root-k64-v32.page.v1";
            break;
        default: throw std::logic_error("Unsupported KVMem native KV storage");
        }
        d.layout = {layout_name, 2, page_tokens, page_bytes, 256, {}};
        const auto& config = program.parameters.model.config().text;
        const std::uint32_t planes = program.kv_storage == KvCacheStorage::BFloat16 ? 2 : 4;
        const kvmem::PayloadPlaneKind kinds[] = {kvmem::PayloadPlaneKind::Key,
            kvmem::PayloadPlaneKind::Value, kvmem::PayloadPlaneKind::KeyScale,
            kvmem::PayloadPlaneKind::ValueScale};
        std::size_t ordinal = 0;
        const auto& cache = program.decoder->text_kv;
        if (cache.position_ranks() > 1)
            throw std::logic_error("KVMem native pages require heads attention");
        const auto shares = cache.head_shares();
        const auto ranks = shares.empty() ? 1U : shares.size();
        std::uint32_t attention_layer = 0;
        for (std::uint32_t layer = 0; layer < config.layer_types.size(); ++layer) {
            if (config.layer_types[layer] != MixerKind::FullAttention) { continue; }
            std::uint32_t head_begin = 0;
            for (std::size_t rank = 0; rank < ranks; ++rank) {
                const auto head_count = static_cast<std::uint32_t>(
                    shares.empty() ? config.attention->num_key_value_heads : shares[rank]);
                const auto owner = static_cast<std::uint32_t>(
                    shares.empty() ? cache.layer_rank(attention_layer) : rank);
                for (std::uint32_t kind = 0; kind < planes; ++kind) {
                    const auto& plane = layout.planes.at(ordinal++);
                    d.layout.planes.push_back({layer, kinds[kind], plane.offset,
                        plane.page_payload_bytes, 1, 0, owner, head_begin, head_count});
                }
                head_begin += head_count;
            }
            if (head_begin != config.attention->num_key_value_heads)
                throw std::logic_error("KVMem page inventory does not cover every KV head");
            ++attention_layer;
        }
        if (ordinal != layout.planes.size()) { throw std::logic_error("KVMem plane inventory mismatch"); }
        if (mtp_layout) {
            for (std::uint32_t kind = 0; kind < planes; ++kind) {
                const auto& plane = mtp_layout->planes.at(kind);
                d.layout.planes.push_back({static_cast<std::uint32_t>(config.layer_types.size()),
                    kinds[kind], layout.page_stride + plane.offset, plane.page_payload_bytes, 1, 0,
                    0, 0, static_cast<std::uint32_t>(config.attention->num_key_value_heads)});
            }
        }
        d.capabilities = {page_tokens, program.capacity, true, true, true, false, false, false};
        return d;
    }

    kvmem::MemorySnapshot snapshot() const override {
        kvmem::MemorySnapshot s;
        s.stamp = stamp();
        const auto count = kv_pages_for_tokens(sequence.text_kv_valid);
        for (std::uint32_t id = 0; id < count; ++id) {
            const auto columns = valid_columns(id, sequence.text_kv_valid);
            s.blocks.push_back({{s.stamp.session, id, id * page_tokens, columns, columns, 0},
                sequence.window.host_versions.at(id)});
        }
        for (auto id : sequence.window.pages) {
            if (id < count) { s.resident.push_back(id); }
        }
        return s;
    }
    kvmem::MemoryStamp stamp() const noexcept override { return sequence.window.stamp; }
    std::unique_ptr<kvmem::MemoryTransfer> prepare(const kvmem::WorkingSetPlan&) override;
    void invalidate() noexcept override { sequence.window.stamp.valid = false; }

    ProgramImpl& program;
    SequenceState& sequence;
    std::uint32_t end;
    bool resident_native = false; // materialization owns RAM leases through restore
    HostKVPageLayout layout;
    std::optional<HostKVPageLayout> mtp_layout;
    std::size_t page_bytes = 0;
};

class WindowMemoryTransfer final : public kvmem::MemoryTransfer {
public:
    WindowMemoryTransfer(WindowMemoryBackend& backend, const kvmem::WorkingSetPlan& plan,
                         bool verify_restore = true)
        : backend(backend), plan(plan), restore_enabled(verify_restore),
          asynchronous(backend.program.memory_io && backend.program.kvmem_options.disk_bytes &&
                       verify_restore && !backend.resident_native) {
        auto& seq = backend.sequence;
        auto& p = backend.program;
        const auto count = kv_pages_for_tokens(seq.text_kv_valid);
        pending.resize(count);
        // Rebuilding a native compact address space also moves retained pages.
        // Reserve their host records up front and include them in physical H and logical history admission.
        // Complete host copies are reused; only the changed tail is copied again.
        std::uint64_t bytes = seq.window.payload_bytes;
        for (std::uint32_t slot = 0; slot < seq.window.pages.size(); ++slot) {
            const auto id = seq.window.pages[slot];
            if (id >= count) { continue; }
            if (!seq.window.archive[id]) {
                if (bytes > p.memory_retained_budget ||
                    backend.page_bytes > p.memory_retained_budget - bytes) {
                    throw kvmem::MemoryError(kvmem::MemoryErrorCode::BudgetExceeded,
                        "KVMem retained payload budget exhausted before transfer");
                }
                bytes += backend.page_bytes;
                if (!asynchronous) pending[id] = p.memory_payload_storage.allocate(backend.page_bytes);
            }
            if (seq.window.host_versions[id] != valid_columns(id, seq.text_kv_valid)) {
                spill_ids.push_back(id);
                spill_slots.push_back(slot);
                sources.push_back(p.text_kv_addresses->physical_page(seq.kv->text, slot));
                source_epochs.push_back(p.text_kv_addresses->content_epoch(seq.kv->text, slot));
                if (backend.mtp_layout) {
                    mtp_sources.push_back(p.backend_kv_addresses->physical_page(*seq.kv->backend, slot));
                    mtp_source_epochs.push_back(p.backend_kv_addresses->content_epoch(*seq.kv->backend, slot));
                }
            }
        }
        prepared_payload_bytes = bytes;
        for (const auto& block : plan.next.blocks) {
            const auto id = static_cast<std::uint32_t>(block.block.id);
            if (!record(id) && (!asynchronous || std::find(spill_ids.begin(), spill_ids.end(), id) == spill_ids.end()))
                throw std::logic_error("KVMem selected page has no native payload");
            selected.push_back(id);
        }
        if (seq.text_kv_valid % page_tokens &&
            (selected.empty() || selected.back() != seq.text_kv_valid / page_tokens)) {
            throw std::logic_error("KVMem partial append page must remain the last selected page");
        }
        const auto rows = std::max(spill_ids.size(), restore_enabled ? selected.size() : 0U);
        if (rows > p.kvmem_window_tokens / page_tokens) {
            throw std::logic_error("KVMem staging exceeds its physical window");
        }
        // One Program-owned buffer is shared by every batch and lane. Transfers
        // synchronize before returning, so no pointer survives a scheduler yield.
        // Grow only to the batch limit. Small captures need no fixed 64 MiB floor.
        if (rows) {
            const auto pages = std::min<std::size_t>(rows,
                std::max<std::size_t>(1, transfer_staging_limit / backend.page_bytes));
            const auto bytes = pages * backend.page_bytes;
            if (!p.memory_kv_staging || p.memory_kv_staging->size() < bytes) {
                // No batch is in flight here. Drop old pin before allocation so
                // growth never temporarily charges both capacities to the driver.
                p.memory_kv_staging.reset();
                p.memory_kv_staging = std::make_unique<PinnedHostBuffer>(bytes);
            }
        }
        if (verify_restore && p.kvmem_options.verify_transfers) {
            const auto bytes_state = p.state_images->host_layout().image_bytes;
            before = std::make_unique<PinnedHostBuffer>(bytes_state);
            after = std::make_unique<PinnedHostBuffer>(bytes_state);
            std::memset(before->data(), 0, bytes_state);
            std::memset(after->data(), 0, bytes_state);
        }
    }

    void start_spill() override {
        auto& p = backend.program;
        auto& seq = backend.sequence;
        KvmemTransferTrace trace(p.device, p.compute_streams, "d2h", spill_ids.size(),
            p.memory_kv_staging ? p.memory_kv_staging->size() : 0, asynchronous);
        if (before) {
            std::array<std::byte*, 1> rows{static_cast<std::byte*>(before->data())};
            p.state_images->copy_to_host_segments(p.state_store->physical_slot(seq.state.write),
                rows, p.state_images->host_layout().image_bytes, p.compute_streams);
            p.device.synchronize(); // Also complete verification capture when no page needs spilling.
        }
        if (asynchronous) {
            storage_work = std::make_shared<StorageWork>(p.memory_payload_storage, backend.page_bytes);
            auto& work = *storage_work;
            work.records.resize(pending.size());
            work.spill_ids = spill_ids;
            work.selected = selected;
            work.sources.resize(selected.size());
            auto scratch = p.memory_transfer_storage.reserve((spill_ids.size() + selected.size()) * backend.page_bytes);
            work.captured = scratch.allocate(spill_ids.size() * backend.page_bytes);
            work.restored = scratch.allocate(selected.size() * backend.page_bytes);
        }
        // Finish every D2H batch before touching its host bytes or reusing pin.
        // No Device page is unmapped until all source pages have durable copies.
        for (std::size_t begin = 0; begin < spill_ids.size();) {
            const auto count = std::min(batch_pages(), spill_ids.size() - begin);
            auto rows = staging_records(count);
            std::memset(rows.front(), 0, count * backend.page_bytes);
            // Every row is a slot of this one pinned batch. Logical block IDs are not allocation
            // groups: supplying them would disable physical run merging, one copy per page.
            const std::span<const std::uint32_t> record_groups;
            std::vector<std::byte*> mtp_rows;
            if (backend.mtp_layout) {
                mtp_rows = rows;
                for (auto& row : mtp_rows) row += backend.layout.page_stride;
                trace.add_copy(p.backend_kv_pages->physical_pool(),
                    std::span(mtp_sources).subspan(begin, count), mtp_rows, *backend.mtp_layout,
                    record_groups);
            }
            trace.add_copy(p.text_kv_pages->physical_pool(), std::span(sources).subspan(begin, count),
                rows, backend.layout, record_groups);
            trace.begin_copy();
            p.text_kv_pages->physical_pool().copy_to_host_records(
                std::span(sources).subspan(begin, count), rows, record_groups, backend.layout, p.compute_streams);
            if (backend.mtp_layout) {
                p.backend_kv_pages->physical_pool().copy_to_host_records(
                    std::span(mtp_sources).subspan(begin, count), mtp_rows,
                    record_groups, *backend.mtp_layout, p.compute_streams);
            }
            trace.end_copy();
            p.device.synchronize();
            trace.batch_complete();
            for (std::size_t i = 0; i < count; ++i) {
                const auto index = begin + i;
                if (source_epochs[index] != p.text_kv_addresses->content_epoch(seq.kv->text, spill_slots[index]))
                    throw std::logic_error("KVMem source epoch changed during spill");
                if (backend.mtp_layout && mtp_source_epochs[index] !=
                    p.backend_kv_addresses->content_epoch(*seq.kv->backend, spill_slots[index]))
                    throw std::logic_error("KVMem MTP source epoch changed during spill");
                if (asynchronous) {
                    std::memcpy(storage_work->captured.writable_data() + index * backend.page_bytes,
                        rows[i], backend.page_bytes);
                } else {
                    auto* destination = record(spill_ids[index]);
                    destination->write(0, rows[i], backend.page_bytes);
                    if (valid_columns(spill_ids[index], seq.text_kv_valid) == page_tokens) destination->seal();
                }
            }
            trace.host_done();
            begin += count;
        }
        if (asynchronous) {
            auto& work = *storage_work;
            for (auto id : spill_ids)
                if (auto* old = record(id)) old->seal();
            for (std::size_t i = 0; i < selected.size(); ++i) {
                if (std::find(spill_ids.begin(), spill_ids.end(), selected[i]) != spill_ids.end()) continue;
                auto* source = record(selected[i]);
                source->seal();
                work.sources[i] = source->source();
            }
            ticket = p.memory_io->submit(work.captured.size() + work.restored.size(), [work = storage_work] {
                for (std::size_t i = 0; i < work->spill_ids.size(); ++i) {
                    auto& destination = work->records[work->spill_ids[i]];
                    destination = work->storage.allocate(work->stride, work->captured.data() + i * work->stride);
                    destination.seal();
                }
                for (std::size_t i = 0; i < work->selected.size(); ++i) {
                    auto* destination = work->restored.writable_data() + i * work->stride;
                    if (const auto& record = work->records[work->selected[i]]; record)
                        record.read(0, destination, work->stride);
                    else work->sources[i].read(0, destination, work->stride);
                }
            });
            seq.window.pending_window_io = ticket;
            trace.finish(); // The storage worker's asynchronous I/O is outside this CPU call.
            return;
        }
        publish_host();
        trace.finish();
    }

    kvmem::TransferPoll poll() override {
        if (asynchronous && !host_published) {
            if (!ticket.ready()) return {kvmem::TransferStatus::Pending, {}};
            try {
                ticket.wait();
                pending = std::move(storage_work->records);
                publish_host();
            } catch (const std::exception& error) {
                return {kvmem::TransferStatus::Failed, error.what()};
            }
        }
        return {kvmem::TransferStatus::Complete, {}};
    }

    void publish_host() {
        auto& seq = backend.sequence;
        for (std::size_t id = 0; id < pending.size(); ++id) {
            if (pending[id]) { seq.window.archive[id] = std::move(pending[id]); }
        }
        for (auto id : spill_ids) { seq.window.host_versions[id] = valid_columns(id, seq.text_kv_valid); }
        seq.window.payload_bytes = prepared_payload_bytes;
        seq.window.spilled_bytes += spill_ids.size() * backend.page_bytes;
        host_published = true;
    }

    void start_restore() override {
        auto& p = backend.program;
        auto& seq = backend.sequence;
        KvmemTransferTrace trace(p.device, p.compute_streams, "h2d", selected.size(),
            p.memory_kv_staging ? p.memory_kv_staging->size() : 0, asynchronous);
        if (!restore_enabled) throw std::logic_error("KVMem archive-only transfer cannot restore");
        if (asynchronous && !host_published)
            throw std::logic_error("KVMem restored before storage completion");
        // Synchronous restores use RAM-only history or materialization's resident
        // records. Retain and validate those leases before destroying the old
        // view, so batching cannot move a fallible disk read past this boundary.
        std::vector<kvmem::HostKvLease> resident;
        if (!asynchronous) {
            resident.reserve(selected.size());
            for (const auto id : selected) {
                auto lease = record(id)->lease();
                if (!lease.data() || lease.size() != backend.page_bytes)
                    throw std::logic_error("KVMem synchronous restore requires a complete resident page");
                resident.push_back(std::move(lease));
            }
        }
        trace.prepare_done();
        destructive = true;
        p.text_kv_addresses->destructive_truncate(seq.kv->text, 0);
        const auto compact_end = static_cast<std::uint32_t>(plan.next.visible_tokens) + backend.end - seq.text_kv_valid;
        p.text_kv_addresses->ensure_mapped_to_tokens(seq.kv->text, compact_end, p.compute_streams);
        if (backend.mtp_layout) {
            p.backend_kv_addresses->destructive_truncate(*seq.kv->backend, 0);
            p.backend_kv_addresses->ensure_mapped_to_tokens(*seq.kv->backend, compact_end, p.compute_streams);
        }
        std::vector<std::byte> expected;
        for (std::size_t begin = 0; begin < selected.size();) {
            const auto count = std::min(batch_pages(), selected.size() - begin);
            auto rows = staging_records(count);
            trace.prepare_done();
            for (std::size_t i = 0; i < count; ++i) {
                if (asynchronous)
                    std::memcpy(rows[i], storage_work->restored.data() + (begin + i) * backend.page_bytes,
                        backend.page_bytes);
                else std::memcpy(rows[i], resident[begin + i].data(), backend.page_bytes);
            }
            trace.host_done();
            if (before) expected.assign(rows.front(), rows.front() + count * backend.page_bytes);
            const std::vector<const std::byte*> input(rows.begin(), rows.end());
            // As in spill, all rows belong to this batch's single pinned allocation.
            const std::span<const std::uint32_t> record_groups;
            std::vector<DeviceKVPageHandle> destinations, mtp_destinations;
            for (std::size_t i = 0; i < count; ++i)
                destinations.push_back(p.text_kv_addresses->physical_page(seq.kv->text,
                    static_cast<std::uint32_t>(begin + i)));
            auto& pool = p.text_kv_pages->physical_pool();
            std::vector<const std::byte*> mtp_input;
            if (backend.mtp_layout) {
                mtp_input = input;
                for (auto& row : mtp_input) row += backend.layout.page_stride;
                for (std::size_t i = 0; i < count; ++i)
                    mtp_destinations.push_back(p.backend_kv_addresses->physical_page(*seq.kv->backend,
                        static_cast<std::uint32_t>(begin + i)));
                trace.add_copy(p.backend_kv_pages->physical_pool(), mtp_destinations, mtp_input,
                    *backend.mtp_layout, record_groups);
            }
            trace.add_copy(pool, destinations, input, backend.layout, record_groups);
            trace.begin_copy();
            pool.copy_from_host_records(input, record_groups, destinations, backend.layout, p.compute_streams);
            if (backend.mtp_layout) {
                p.backend_kv_pages->physical_pool().copy_from_host_records(mtp_input, record_groups,
                    mtp_destinations, *backend.mtp_layout, p.compute_streams);
            }
            trace.end_copy();
            p.device.synchronize();
            trace.batch_complete();
            if (before) {
                // Verify through the same pin window; expected bytes stay in pageable RAM.
                pool.copy_to_host_records(destinations, rows, {}, backend.layout, p.compute_streams);
                if (backend.mtp_layout) {
                    auto mtp_rows = rows;
                    for (auto& row : mtp_rows) row += backend.layout.page_stride;
                    p.backend_kv_pages->physical_pool().copy_to_host_records(mtp_destinations, mtp_rows,
                        {}, *backend.mtp_layout, p.compute_streams);
                }
                p.device.synchronize();
                if (std::memcmp(rows.front(), expected.data(), expected.size()))
                    throw std::logic_error("KVMem native payload changed during restore");
            }
            begin += count;
        }
        if (before) {
            std::array<std::byte*, 1> state{static_cast<std::byte*>(after->data())};
            p.state_images->copy_to_host_segments(p.state_store->physical_slot(seq.state.write),
                state, p.state_images->host_layout().image_bytes, p.compute_streams);
            p.device.synchronize();
            if (std::memcmp(before->data(), after->data(), p.state_images->host_layout().image_bytes))
                throw std::logic_error("KVMem page transfer changed recurrent state");
        }
        seq.window.restored_bytes += selected.size() * backend.page_bytes;
        trace.finish();
    }

    void publish(const kvmem::ExecutionView& view) override {
        auto& p = backend.program;
        auto& seq = backend.sequence;
        p.text_kv_addresses->commit_frontier(seq.kv->text, static_cast<std::uint32_t>(view.visible_tokens));
        if (backend.mtp_layout) {
            p.backend_kv_addresses->commit_frontier(*seq.kv->backend,
                static_cast<std::uint32_t>(view.visible_tokens));
            // The earlier proposal was generated against the previous view. Rebuild
            // it from the next verified anchor instead of publishing stale drafts.
            seq.mtp_draft_count = 0;
        }
        seq.window.removed_tokens = seq.text_kv_valid - static_cast<std::uint32_t>(view.visible_tokens);
        seq.window.pages = std::move(selected);
        seq.window.stamp = view.stamp;
        ++seq.window.swaps;
        published = true;
        if (p.kvmem_options.verify_transfers) {
            std::fprintf(stderr, "KVMEM_NATIVE_VIEW logical=%u compact=%llu physical_pages=%u host_bytes=%llu swap=%u payload_exact=1 state_exact=1\n",
                seq.text_kv_valid, static_cast<unsigned long long>(view.visible_tokens),
                p.text_kv_pages->physical_pool().allocated_pages(),
                static_cast<unsigned long long>(seq.window.payload_bytes), seq.window.swaps);
        }
    }

    kvmem::AbortResult abort() noexcept override {
        // Cancellation never publishes a partial restore. Drain a started
        // CPU task before the caller may reuse this sequence's old records.
        if (asynchronous) {
            ticket.cancel();
            try { ticket.wait(); } catch (...) {}
        }
        try { backend.program.device.synchronize(); }
        catch (...) { backend.invalidate(); return kvmem::AbortResult::SessionInvalidated; }
        if (destructive && !published) {
            backend.invalidate();
            return kvmem::AbortResult::SessionInvalidated;
        }
        return kvmem::AbortResult::PreviousViewPreserved;
    }

private:
    struct StorageWork {
        StorageWork(const kvmem::HostKvStorage& storage, std::size_t stride) : storage(storage), stride(stride) {}
        kvmem::HostKvStorage storage;
        std::size_t stride;
        std::vector<std::uint32_t> spill_ids, selected;
        std::vector<KvmemHostRecord> records;
        std::vector<kvmem::HostKvSource> sources;
        KvmemHostRecord captured, restored;
    };
    std::size_t batch_pages() const {
        const auto& buffer = backend.program.memory_kv_staging;
        if (!buffer || buffer->size() < backend.page_bytes)
            throw std::logic_error("KVMem transfer staging cannot hold one native page");
        return buffer->size() / backend.page_bytes;
    }
    std::vector<std::byte*> staging_records(std::size_t count) const {
        auto* staging = static_cast<std::byte*>(backend.program.memory_kv_staging->data());
        std::vector<std::byte*> records;
        records.reserve(count);
        for (std::size_t i = 0; i < count; ++i) records.push_back(staging + i * backend.page_bytes);
        return records;
    }
    KvmemHostRecord* record(std::uint32_t id) {
        if (pending.at(id)) return &pending[id];
        auto & archived = backend.sequence.window.archive.at(id);
        return archived ? &archived : nullptr;
    }
    WindowMemoryBackend& backend;
    kvmem::WorkingSetPlan plan;
    std::vector<KvmemHostRecord> pending;
    std::vector<std::uint32_t> spill_ids, spill_slots, selected;
    std::vector<DeviceKVPageHandle> sources, mtp_sources;
    std::vector<std::uint64_t> mtp_source_epochs;
    std::vector<std::uint64_t> source_epochs;
    std::unique_ptr<PinnedHostBuffer> before, after;
    std::uint64_t prepared_payload_bytes = 0;
    const bool restore_enabled;
    const bool asynchronous;
    bool host_published = false;
    std::shared_ptr<StorageWork> storage_work;
    kvmem::StorageTicket ticket;
    bool destructive = false, published = false;
};

std::unique_ptr<kvmem::MemoryTransfer> WindowMemoryBackend::prepare(const kvmem::WorkingSetPlan& plan) {
    if (!(plan.expected == stamp())) { throw std::logic_error("KVMem plan is stale"); }
    return std::make_unique<WindowMemoryTransfer>(*this, plan);
}

std::vector<std::uint32_t> ProgramImpl::memory_image_pages(const PreparedPromptData& prompt) const {
    std::vector<std::uint32_t> pages;
    for (const auto& item : prompt.vision_items) {
        if (item.modality != PromptModality::Image) {
            throw RequestError(RequestErrorKind::InvalidMedia, "KVMem currently supports images, not video");
        }
        for (const auto& span : item.token_spans) {
            if (!span.count || span.begin > prompt.token_ids.size() ||
                span.count > prompt.token_ids.size() - span.begin) {
                throw std::logic_error("KVMem received an invalid image token span");
            }
            const auto last = (span.begin + span.count - 1) / page_tokens;
            for (auto id = span.begin / page_tokens; id <= last; ++id)
                pages.push_back(static_cast<std::uint32_t>(id));
        }
    }
    std::sort(pages.begin(), pages.end());
    pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
    if (!pages.empty()) {
        std::uint32_t extra_sink = 0;
        for (std::uint32_t id = 0; id < kvmem_options.sink_blocks(); ++id) {
            if (!std::binary_search(pages.begin(), pages.end(), id)) ++extra_sink;
        }
        // Recent pages include the mandatory partial append tail; startup requires at least one.
        const auto tail_pages = kvmem_options.recent_blocks();
        if (pages.size() + extra_sink + tail_pages > kvmem_options.selected_tokens / page_tokens) {
            throw RequestError(RequestErrorKind::ContextLengthExceeded,
                "KVMem selection budget cannot retain complete image regions; increase the budget or reduce image resolution");
        }
    }
    return pages;
}

bool ProgramImpl::prepare_memory_append(SequenceState& sequence, std::uint32_t begin,
                                       std::uint32_t end, bool allow_pending) {
    if (!memory_io || begin == end) return true;
    // A legal short/non-reusable request may have no Host payload reservation.
    if (!kvmem_options.disk_bytes && !requests.at(sequence.lane).memory_payload_reservation) return true;
    auto& window = sequence.window;
    if (window.pending_append) {
        auto& preparation = *window.pending_append;
        if (!preparation.ticket.ready()) {
            if (allow_pending) return false;
            preparation.ticket.wait();
        }
        preparation.ticket.wait();
        auto& work = *preparation.work;
        for (std::size_t i = 0; i < work.ids.size(); ++i) {
            auto& destination = window.archive.at(work.ids[i]);
            if (!destination) window.payload_bytes += work.records[i].size();
            destination = std::move(work.records[i]);
        }
        window.pending_append.reset();
    }
    const auto count = kv_pages_for_tokens(end);
    if (window.archive.size() < count) window.archive.resize(count);
    if (window.host_versions.size() < count) window.host_versions.resize(count);
    const auto stride = text_host_kv_page_stride + backend_host_kv_page_stride;
    // Speculative device mappings can extend beyond the request's output limit.
    // RAM lookahead may prepare only the history admitted for this request;
    // those extra device positions never need durable Host records.
    const auto prepare_count = kvmem_options.disk_bytes ? count :
        std::min<std::uint64_t>(count, requests.at(sequence.lane).memory_payload_reservation / stride);
    std::shared_ptr<KvmemAppendPreparation::Work> work;
    for (auto id = begin / page_tokens; id < prepare_count; ++id) {
        const auto& record = window.archive[id];
        // Preserve the existing RAM owner-borrow/write path, without speculative COW.
        if (!kvmem_options.disk_bytes && record) continue;
        if (record.mutable_resident()) continue;
        if (!work) work = std::make_shared<KvmemAppendPreparation::Work>();
        work->ids.push_back(id);
        work->sources.push_back(record ? record.source() : kvmem::HostKvSource{});
    }
    if (!work) return true;
    work->records.resize(work->ids.size());
    auto preparation = std::make_unique<KvmemAppendPreparation>();
    preparation->work = work;
    preparation->ticket = memory_io->submit(work->ids.size() * stride,
        [work, storage = memory_payload_storage, stride]() mutable {
            for (std::size_t i = 0; i < work->ids.size(); ++i) {
                auto& destination = work->records[i];
                destination = storage.allocate(stride);
                if (work->sources[i]) work->sources[i].read(0, destination.writable_data(), stride);
            }
            work->sources.clear();
        });
    window.pending_append = std::move(preparation);
    if (allow_pending) return false;
    return prepare_memory_append(sequence, begin, end, false);
}

bool ProgramImpl::prepare_window(SequenceState& sequence, std::uint32_t begin, std::uint32_t end,
                                  bool force_selection, bool allow_pending, kvmem::SelectionReason reason) {
    if (!kvmem_window_tokens) { return true; }
    if (!sequence.kv || begin != sequence.text_kv_valid || end < begin || end > capacity ||
        end - begin > kvmem_options.reserve_tokens) {
        throw std::logic_error("KVMem append exceeds logical frontier or reserve");
    }
    auto& pending = sequence.window.pending_window;
    if (pending) {
        if (sequence.window.pending_window_begin != begin ||
            (end > sequence.window.pending_window_end && sequence.window.pending_window_end != begin))
            throw std::logic_error("KVMem pending window no longer matches its request boundary");
        for (;;) {
            const auto epoch = memory_io->completion_epoch();
            const auto phase = pending->advance();
            if (phase == kvmem::MemoryPhase::Ready) break;
            if (phase == kvmem::MemoryPhase::Spilling) {
                if (allow_pending) { sequence.window.pending_io_epoch = epoch; return false; }
                memory_io->wait_for_completion(epoch, std::chrono::milliseconds(1));
            }
        }
        pending->publish();
        pending.reset();
        force_selection = false;
    }
    if (end == begin && !force_selection) { return true; }
    if (sequence.window.query_replay_active && !force_selection &&
        compact_position(sequence,end)>kvmem_window_tokens)
        throw std::logic_error("bounded Query replay cannot execute Pressure");
    if (force_selection || compact_position(sequence, end) > kvmem_window_tokens) {
        // A reselection replaces the selected blocks; only the trailing recent pages are
        // guaranteed to stay. Raise the masked-draft floor to that trailing continuous segment
        // so the DFlash2 ring stops reading features of blocks that were swapped out.
        const std::uint64_t recent_tokens =
            static_cast<std::uint64_t>(kvmem_options.recent_blocks()) * page_tokens;
        sequence.dflash_context_floor =
            end > recent_tokens ? static_cast<std::uint32_t>(end - recent_tokens) : 0U;
        auto backend = std::make_shared<WindowMemoryBackend>(*this, sequence, end);
        const auto count = kv_pages_for_tokens(begin);
        const auto keep = std::min(count, kvmem_options.selected_tokens / page_tokens);
        auto& state = sequence.window;
        std::vector<std::uint64_t> mandatory;
        if (count == 0) {
            mandatory.push_back(0);
        } else {
            const auto sink_keep = std::min(kvmem_options.sink_blocks(), count);
            for (std::uint32_t id = 0; id < sink_keep; ++id) mandatory.push_back(id);
            for (const auto id : state.media_pages) if (id < count) mandatory.push_back(id);
            const auto recent_keep = std::min(kvmem_options.recent_blocks(), count);
            for (std::uint32_t i = 0; i < recent_keep; ++i) mandatory.push_back(count - 1 - i);
        }
        std::sort(mandatory.begin(), mandatory.end());
        mandatory.erase(std::unique(mandatory.begin(), mandatory.end()), mandatory.end());
        std::vector<std::uint64_t> selected;
        const bool query_complete = state.query_tokens != 0 &&
            (state.query_replayed || begin >= state.query_end) &&
            state.query_tokens == state.query_end - state.query_begin;
        if (reason==kvmem::SelectionReason::Query && query_complete) {
            if (!state.selector || state.statistics_frontier != begin ||
                state.selector->total_tokens() != begin) {
                throw std::logic_error("KVMem retrieval index does not match the evaluated prefix");
            }
            if (state.query_selected.empty())
                throw std::logic_error("Query selection must be admitted before native restore");
            const auto& ids=state.query_selected;
            selected.assign(ids.begin(),ids.end());
            mandatory.assign(state.query_required.begin(),state.query_required.end());
            if (kvmem_options.verify_transfers) {
                std::size_t cold = 0;
                for (auto id : ids) {
                    if (std::find(state.pages.begin(), state.pages.end(), id) == state.pages.end()) ++cold;
                }
                std::fprintf(stderr, "KVMEM_RETRIEVAL logical=%u query_tokens=%u cold=%zu selected=",
                    begin, state.query_tokens, cold);
                for (std::size_t i = 0; i < ids.size(); ++i) std::fprintf(stderr, "%s%u", i ? "," : "", ids[i]);
                std::fprintf(stderr, "\n");
            }
        } else {
            std::vector<std::uint32_t> candidates;
            std::vector<std::uint64_t> starts;
            for(auto id:state.pages) if(id<count) { candidates.push_back(id); starts.push_back(std::uint64_t(id)*64); }
            const std::vector<std::uint32_t> required(mandatory.begin(),mandatory.end());
            for(auto id:required) if(std::find(candidates.begin(),candidates.end(),id)==candidates.end() && count)
                throw std::logic_error("KVMem pressure cannot recall a cold mandatory page");
            const auto scores=state.feedback.pressure_scores(starts);
            const auto ids=kvmem::pressure_selection(candidates,scores,required,std::max(1u,keep));
            selected.assign(ids.begin(),ids.end());
            if(kvmem_options.verify_transfers) std::fprintf(stderr,
                "KVMEM_PRESSURE phase=%s logical=%u candidates=%zu selected=%zu graph=0\n",
                reason==kvmem::SelectionReason::DecodePressure?"decode":"prefill",begin,candidates.size(),ids.size());
        }
        const kvmem::MemoryBudget budget{kvmem_options.selected_tokens, kvmem_options.reserve_tokens,
            kvmem_window_tokens, memory_retained_budget};
        if (memory_io && kvmem_options.disk_bytes) {
            auto session = std::make_unique<kvmem::MemorySession>(backend);
            session->prepare(backend->stamp(), selected, mandatory, budget);
            sequence.window.pending_window_begin = begin;
            sequence.window.pending_window_end = end;
            pending = std::move(session);
            return prepare_window(sequence, begin, end, false, allow_pending, reason);
        }
        kvmem::MemorySession session(backend);
        session.prepare(backend->stamp(), selected, mandatory, budget);
        while (session.advance() != kvmem::MemoryPhase::Ready) {}
        session.publish();
    }
    // Any host allocation/promotion that might spill belongs to the I/O job.
    // Checkpoint capture after this chunk only writes already-resident buffers.
    if (!prepare_memory_append(sequence, begin, end, allow_pending)) return false;
    ensure_sequence_kv_mapped(sequence, end, backend_kv_pages ? end : 0);
    const auto mapped = kv_pages_for_tokens(compact_position(sequence, end));
    while (sequence.window.pages.size() < mapped) {
        const auto slot = static_cast<std::uint32_t>(sequence.window.pages.size());
        sequence.window.pages.push_back(slot + sequence.window.removed_tokens / page_tokens);
    }
    if (text_kv_pages->physical_pool().allocated_pages() >
        static_cast<std::uint64_t>(max_concurrency) * kvmem_window_tokens / page_tokens) {
        throw std::logic_error("KVMem exceeded its native physical page budget");
    }
    return true;
}
bool ProgramImpl::memory_pending(SequenceHandle handle) const noexcept {
    if (!memory_io || !valid_sequence(handle)) return false;
    const auto& window = active_sequence(ContractAccess::lane(handle).value).window;
    if (window.pending_append && !window.pending_append->ticket.ready()) return true;
    return window.pending_window && window.pending_io_epoch == memory_io->completion_epoch() &&
        memory_io->pending_jobs() != 0;
}

bool ProgramImpl::prepare_decode_memory(SequenceHandle handle) {
    if (!memory_io || !kvmem_options.disk_bytes) return true;
    if (!valid_sequence(handle)) throw std::logic_error("KVMem decode preparation has a stale sequence");
    if (memory_pending(handle)) return false;
    auto& sequence = active_sequence(ContractAccess::lane(handle).value);
    const auto append = backend_kv_pages ? std::max(draft_window, ngram_draft_window) + draft_window : 1U;
    return prepare_window(sequence, sequence.execution_frontier,
        std::min(capacity, sequence.execution_frontier + append), false, true,kvmem::SelectionReason::DecodePressure);
}
std::uint32_t ProgramImpl::memory_control_chunk() const noexcept {
    return memory_io && kvmem_options.disk_bytes
        ? std::min(prefill_chunk, kvmem_options.reserve_tokens) : UINT32_MAX;
}
bool ProgramImpl::prepare_control_memory(SequenceHandle handle, std::uint32_t tokens) {
    if (!memory_io || !kvmem_options.disk_bytes) return true;
    if (!valid_sequence(handle)) throw std::logic_error("KVMem control preparation has a stale sequence");
    if (memory_pending(handle)) return false;
    auto& sequence = active_sequence(ContractAccess::lane(handle).value);
    return prepare_window(sequence, sequence.execution_frontier,
                          sequence.execution_frontier + tokens, false, true,kvmem::SelectionReason::DecodePressure);
}
bool ProgramImpl::cancel_memory_ready(SequenceHandle handle) noexcept {
    if (!memory_io || !valid_sequence(handle)) return true;
    auto& window = active_sequence(ContractAccess::lane(handle).value).window;
    if (window.pending_append) window.pending_append->ticket.cancel();
    window.pending_window_io.cancel();
    // Keep the sequence parked while an already-started CPU task drains.
    // Other lanes remain runnable; abort will only encounter ready tickets.
    return (!window.pending_append || window.pending_append->ticket.ready()) &&
           window.pending_window_io.ready();
}
// Capture uses the same complete-before-publish Host spill as a working-set
// transfer, but stops before any destructive restore. Native device ownership
// and attention view remain unchanged.
void ProgramImpl::archive_memory_window(SequenceState& sequence) {
    WindowMemoryBackend backend(*this, sequence, sequence.text_kv_valid);
    const auto snapshot = backend.snapshot();
    kvmem::WorkingSetPlan plan;
    plan.expected = snapshot.stamp;
    for (auto id : snapshot.resident) {
        const auto& block = snapshot.blocks.at(static_cast<std::size_t>(id));
        plan.next.blocks.push_back({block.block, plan.next.visible_tokens});
        plan.next.visible_tokens += block.block.valid_tokens;
    }
    WindowMemoryTransfer spill(backend, plan, false);
    try {
        spill.start_spill();
    } catch (...) {
        (void)spill.abort();
        throw;
    }
}

bool ProgramImpl::memory_query_probe(const SequenceState& sequence, std::uint32_t prompt_tokens) const {
    const auto& window = sequence.window;
    const auto workspace=backend_kv_pages ? std::max(draft_window,ngram_draft_window)+draft_window : 0;
    const auto admission=kvmem::query_replay_plan({window.query_begin,window.query_end},
        window.query_begin,prompt_tokens,kvmem_options.query_replay_max_tokens,page_tokens,
        std::min(kv_pages_for_tokens(window.query_begin),kvmem_options.selected_tokens/page_tokens),
        kvmem_window_tokens/page_tokens,workspace);
    // A small logical prompt is not proof of a complete native view: an earlier
    // draft workspace can already have caused Pressure below B+R.
    const bool needs_recall=window.removed_tokens != 0 ||
        std::uint64_t(prompt_tokens)+workspace>kvmem_window_tokens;
    return kvmem_window_tokens && needs_recall &&
        window.query_begin > 0 && !window.query_replayed && admission.admitted;
}


void ProgramImpl::progress_checkpoint_coverage(KvmemWindowState& window, bool force) {
    if(!window.coverage.policy().enabled) return;
    std::vector<kvmem::CheckpointInfo> fixed_info;
    for(const auto& record:window.checkpoints) fixed_info.push_back(record.checkpoint_info());
    window.coverage.prune(fixed_info);
    const auto fixed = [&](const KvmemPrefixCheckpoint* cp) {
        for(const auto& lane:continuation_states) for(const auto kind:kKvmemCheckpointKinds)
            if(memory_checkpoint(lane.window,kind)==cp) return true;
        return false;
    };
    if(memory_coverage_transfer) {
        auto& pending=*memory_coverage_transfer;
        if(!pending.ticket.ready()) return;
        bool owned=false;
        const auto contains=[&](const KvmemWindowState& owner) {
            if(!owner.stamp.valid || owner.stamp.session != pending.checkpoint->identity.stamp.session) return;
            for(const auto& entry:owner.coverage) if(entry.info.identity==pending.checkpoint.get()) owned=true;
        };
        for(const auto& lane:continuation_states) contains(lane.window);
        for(const auto& history:memory_histories) if(history) contains(*history);
        if(pending.ticket.status()==kvmem::StorageTaskStatus::Complete && owned &&
                pending.checkpoint->state==pending.source && !fixed(pending.checkpoint.get())) {
            pending.checkpoint->cold_state=std::move(pending.destination);
            pending.checkpoint->cold_payload_bytes=pending.source->size();
            pending.checkpoint->cold_tag=uint64_t(reinterpret_cast<uintptr_t>(pending.checkpoint.get()));
            pending.checkpoint->state.reset();
            if(kvmem_options.verify_transfers) std::fprintf(stderr,"KVMEM_COVERAGE_TIER write_bytes=%zu\n",pending.source->size());
        }
        memory_coverage_transfer.reset();
    }
    const auto disk=memory_payload_storage.spill_file();
    if(!disk || !memory_io || memory_idle_transfer) return;
    std::vector<const KvmemPrefixCheckpoint*> hot;
    for(const auto kind:kKvmemCheckpointKinds) if(const auto* cp=memory_checkpoint(window,kind); cp && cp->state)
        if(std::find(hot.begin(),hot.end(),cp)==hot.end()) hot.push_back(cp);
    const kvmem::CheckpointArchive<KvmemCheckpointRecord>::Entry* victim=nullptr;
    for(const auto& entry:window.coverage) if(entry.info.identity && entry.record.data && entry.record.data->state) {
        if(std::find(hot.begin(),hot.end(),entry.record.data.get())==hot.end()) hot.push_back(entry.record.data.get());
        if(!fixed(entry.record.data.get()) && (!victim || entry.info.used<victim->info.used)) victim=&entry;
    }
    if(!victim || (!force && hot.size()<=window.coverage.policy().hot_count)) return;
    try {
        auto pending=std::make_unique<KvmemCoverageTransfer>();
        pending->checkpoint=victim->record.data;
        pending->source=pending->checkpoint->state;
        const auto bytes=pending->source->size();
        pending->destination=disk->reserve(kvmem::checkpoint_image_size(bytes,1));
        const auto destination=pending->destination;
        const auto tag=uint64_t(reinterpret_cast<uintptr_t>(pending->checkpoint.get()));
        // The engine owns the pinned allocation; worker only borrows its bytes.
        const auto data=pending->source->data();
        pending->ticket=memory_io->submit(bytes,[destination,tag,data,bytes] {
            kvmem::write_checkpoint_image(destination,tag,{{data,bytes}});
        });
        if(pending->ticket) memory_coverage_transfer=std::move(pending);
    } catch(const std::exception& error) {
        std::fprintf(stderr,"KVMEM_COVERAGE_TIER retained RAM: %s\n",error.what());
    }
}

void ProgramImpl::checkpoint_memory_room(KvmemWindowState& window,size_t bytes,bool optional) {
    const auto& policy=window.coverage.policy();
    if(!policy.enabled) return;
    const auto budget=policy.host_bytes*max_concurrency;
    // Reserve the single bounded worker read buffer even before it starts.
    const auto staging=kvmem_options.disk_bytes ? state_images->host_layout().image_bytes : 0;
    const auto fits=[&] {
        const auto used=memory_summary().kvmem.checkpoint_bytes+staging;
        return used<=budget && bytes<=budget-used;
    };
    if(fits()) return;
    memory_checkpoint_pool->trim();
    if(!optional) {
        if(memory_idle_transfer) {
            // The idle and active paths share the one checkpoint writer. A
            // required capture may drain it, without immediately starting a
            // second write that would keep the released pinned image leased.
            const bool requested=std::exchange(memory_idle_requested,false);
            try { memory_idle_transfer->ticket.wait(); } catch(...) {}
            (void)progress_memory_storage();
            memory_idle_requested=memory_idle_requested || requested;
            memory_checkpoint_pool->trim();
        }
        if(memory_coverage_transfer) { try { memory_coverage_transfer->ticket.wait(); } catch(...) {} }
        progress_checkpoint_coverage(window);
        while(!fits()) {
            progress_checkpoint_coverage(window,true);
            if(!memory_coverage_transfer) break;
            const auto before=memory_checkpoint_pool->allocated_bytes();
            try { memory_coverage_transfer->ticket.wait(); } catch(...) {}
            progress_checkpoint_coverage(window);
            memory_checkpoint_pool->trim();
            if(memory_checkpoint_pool->allocated_bytes()>=before) break;
        }
        // The admitted restore source is an execution lease, including its
        // archive directory entry while the caller holds a slot reference.
        uint64_t restoring=0;
        if(const auto* transaction=std::get_if<MaterializationTransaction>(&context_transaction_)) {
            const auto& staged=requests[transaction->destination.value].prefill;
            if(staged) restoring=staged->memory_restore_generation;
        }
        const auto reclaim=[&](KvmemWindowState& owner) {
            if(restoring && owner.stamp.session.id==restoring) return;
            for(auto& entry:owner.coverage) if(entry.info.identity && !fits()) {
                bool held=false;
                for(const auto kind:kKvmemCheckpointKinds) held|=memory_checkpoint(owner,kind)==entry.record.data.get();
                if(!held) entry={};
                memory_checkpoint_pool->trim();
            }
        };
        for(auto& lane:continuation_states) reclaim(lane.window);
        for(auto& history:memory_histories) if(history) reclaim(*history);
        // A fresh conversation may need the quota held by an older idle owner.
        // The currently admitted restore source remains protected until publish.
        while(!fits()) {
            std::optional<size_t> victim;
            for(size_t i=0;i<memory_histories.size();++i) if(memory_histories[i] &&
                    memory_histories[i]->stamp.session.id!=restoring && (!victim ||
                    memory_histories[i]->publication_order<memory_histories[*victim]->publication_order)) victim=i;
            if(!victim) break;
            memory_histories[*victim].reset(); ++memory_history_evictions;
            memory_checkpoint_pool->trim();
        }
    }
    if(!fits()) throw std::runtime_error("checkpoint Host budget exhausted");
}

bool ProgramImpl::skip_memory_coverage(const SequenceState& sequence) const {
    const auto* earlier=sequence.window.coverage.nearby(sequence.text_kv_valid,[&](const auto& record) {
        const auto& cp=*record.data;
        const auto frontier=cp.prefix.size();
        return cp.state_bytes() && frontier==cp.identity.stamp.frontier &&
            cp.identity.stamp.session==sequence.window.stamp.session &&
            cp.prefix_identity.prefix_equals(sequence.prefix_identity,frontier);
    });
    if(!earlier) return false;
    if(kvmem_options.verify_transfers) std::fprintf(stderr,"KVMEM_CHECKPOINT_NEARBY frontier=%u earlier=%zu\n",
        sequence.text_kv_valid,earlier->data->prefix.size());
    return true;
}

void ProgramImpl::capture_memory_coverage(SequenceState& sequence,uint32_t chunk_tokens) noexcept {
    auto& window=sequence.window;
    if(!window.coverage.policy().enabled || !chunk_tokens) return;
    try {
        progress_checkpoint_coverage(sequence.window);
        const auto frontier=sequence.text_kv_valid;
        const auto purpose=window.capture_plan.due(frontier);
        if(!purpose) return;
        window.capture_plan.saved(frontier);
        const auto& request=requests[sequence.lane];
        if(request.prefill) for(const auto& item:request.prefill->prompt.vision_items)
            if(!item.token_spans.empty() && item.token_spans.front().begin<frontier &&
                frontier<item.token_spans.back().begin+item.token_spans.back().count) return;
        if(skip_memory_coverage(sequence)) return;
        const auto bytes=state_images->host_layout().image_bytes;
        // Keep one further native image available for required execution state.
        checkpoint_memory_room(sequence.window,2*bytes,true);
        copy_tail(sequence,prefill_hidden.slice(1,int32_t(chunk_tokens)-1,1));
        std::shared_ptr<KvmemPrefixCheckpoint> checkpoint=capture_memory_checkpoint(sequence,true);
        const bool kept=window.coverage.remember({kvmem::CheckpointPurpose::None,checkpoint},*purpose);
        if(kvmem_options.verify_transfers) std::fprintf(stderr,"KVMEM_COVERAGE frontier=%u retained=%d count=%zu\n",frontier,int(kept),window.coverage.size());
        progress_checkpoint_coverage(sequence.window);
    } catch(const std::exception& error) {
        std::fprintf(stderr,"KVMEM_COVERAGE capture skipped: %s\n",error.what());
    }
}

std::uint32_t ProgramImpl::memory_checkpoint_frontier(const SequenceState& sequence,
    std::uint32_t prompt_tokens, bool allow_reuse) const {
    if (memory_query_probe(sequence, prompt_tokens)) return sequence.window.query_begin;
    // Leave one suffix token to evaluate and sample normally, including the MTP
    // boundary row rebuilt on restore. No query probe/replay is needed in this path.
    if (kvmem_window_tokens && allow_reuse && prompt_tokens > 1 &&
        prompt_tokens <= kvmem_window_tokens) return prompt_tokens - 1;
    return 0;
}

bool ProgramImpl::complete_memory_checkpoint(const SequenceState& sequence,
                                             const KvmemPrefixCheckpoint& checkpoint) const {
    const auto& window = sequence.window;
    const auto frontier = checkpoint.prefix.size();
    // These records were captured by, or natively admitted into, this Program's
    // window. The session binds their backend/layout; restore still verifies the
    // full descriptor before mutation. This lookup performs no I/O or allocation.
    if (!window.stamp.valid || !frontier || frontier > sequence.text_kv_valid ||
        frontier > sequence.ledger.size() || !window.statistics || !window.selector ||
        frontier > window.statistics_frontier || frontier != checkpoint.identity.stamp.frontier ||
        !(checkpoint.identity.stamp.session == window.stamp.session) ||
        checkpoint.state_bytes() != state_images->host_layout().image_bytes ||
        checkpoint.graph.accepted_through != frontier || checkpoint.graph.block_tokens != page_tokens ||
        !std::equal(checkpoint.prefix.begin(), checkpoint.prefix.end(), sequence.ledger.begin()) ||
        !checkpoint.prefix_identity.prefix_equals(sequence.prefix_identity, frontier)) return false;
    const auto pages = kv_pages_for_tokens(static_cast<std::uint32_t>(frontier));
    if (window.archive.size() < pages || window.host_versions.size() < pages) return false;
    for (std::uint32_t page = 0; page < pages; ++page)
        if (!window.archive[page] || window.host_versions[page] < valid_columns(page, frontier)) return false;
    return std::all_of(checkpoint.resident_pages.begin(), checkpoint.resident_pages.end(),
                       [pages](auto page) { return page < pages; });
}

bool ProgramImpl::skip_memory_tail_checkpoint(SequenceState& sequence, std::uint32_t prompt_tokens) {
    auto& window = sequence.window;
    const auto limit = window.coverage.policy().extra_recompute_tokens;
    if (!limit || !kvmem_window_tokens || prompt_tokens <= 1 ||
        prompt_tokens > kvmem_window_tokens || memory_query_probe(sequence, prompt_tokens)) return false;
    const auto frontier = prompt_tokens - 1;
    if (const auto* existing = memory_checkpoint(window, KvmemCheckpointKind::PromptTail);
        existing && existing->prefix.size() == frontier) return false;
    const auto minimum = window.query_end;
    std::shared_ptr<KvmemPrefixCheckpoint> earlier;
    for (const auto kind : checkpoint_kinds(window)) {
        const auto& candidate = memory_checkpoint_slot(window, kind);
        if (!candidate || !checkpoint_recompute_allowed(candidate->prefix.size(), frontier, limit, minimum) ||
            (earlier && candidate->prefix.size() <= earlier->prefix.size()) ||
            !complete_memory_checkpoint(sequence, *candidate)) continue;
        earlier = candidate;
    }
    if (!earlier) return false; // A planned or failed InputFallback is not a saved state.
    auto& lease = window.checkpoints.slot(kvmem::CheckpointPurpose::PromptTail).data;
    const bool changed = lease != earlier;
    lease = std::move(earlier);
    // A fixed slot keeps both directory identity and payload alive for this
    // request, including cancellation, without creating another native image.
    if (changed && kvmem_options.verify_transfers)
        std::fprintf(stderr, "KVMEM_CHECKPOINT_SKIP purpose=prompt_tail frontier=%u earlier=%zu extra_tokens=%zu limit=%llu source_cold=%d\n",
            frontier, lease->prefix.size(), frontier - lease->prefix.size(),
            static_cast<unsigned long long>(limit), int(!lease->state));
    return true;
}

std::uint32_t ProgramImpl::memory_input_reuse_frontier(const PreparedPromptData& prompt) const {
    if (!kvmem_window_tokens || !prompt.identity.rewrite_checkpoint) return 0;
    const auto frontier = structural_input_reuse_frontier(static_cast<std::uint32_t>(prompt.token_ids.size()),
        prompt.memory_input_end, prompt.identity.rewrite_checkpoint->frontier);
    const auto query = memory_query_span(prompt);
    if (!frontier || (query.count && frontier < query.begin + query.count)) return 0;
    for (const auto& item : prompt.vision_items) if (!item.token_spans.empty() &&
        item.token_spans.front().begin < frontier &&
        frontier < item.token_spans.back().begin + item.token_spans.back().count) return 0;
    return frontier;
}

bool ProgramImpl::reuse_memory_input_checkpoint(SequenceState& sequence, std::uint32_t frontier,
                                                std::uint32_t fallback, bool query_protected) {
    auto& window = sequence.window;
    const auto limit = window.coverage.policy().extra_recompute_tokens;
    std::shared_ptr<KvmemPrefixCheckpoint> source;
    for (const auto kind : checkpoint_kinds(window)) {
        const auto& cp = memory_checkpoint_slot(window, kind);
        if (!cp || !input_reuse_source_allowed(cp->prefix.size(), frontier, limit, window.query_end) ||
            (source && cp->prefix.size() <= source->prefix.size()) ||
            !complete_memory_checkpoint(sequence, *cp)) continue;
        source = cp;
    }
    if (!source) return false;
    auto& input = window.checkpoints.slot(kvmem::CheckpointPurpose::InputReuse).data;
    const bool changed = input != source;
    input = std::move(source);
    // This role never retires QueryReplay. Its exact source remains necessary
    // throughout the probe, waiting-for-rewind and active-replay phases.
    if (changed && kvmem_options.verify_transfers)
        std::fprintf(stderr, "KVMEM_INPUT_REUSE action=%s frontier=%u earlier=%zu extra_tokens=%zu limit=%llu fallback=%u query_protected=%d\n",
            input->prefix.size() == frontier ? "share" : "nearby", frontier, input->prefix.size(),
            frontier - input->prefix.size(), static_cast<unsigned long long>(limit), fallback, int(query_protected));
    return true;
}

void ProgramImpl::capture_memory_input_reuse(SequenceState& sequence, std::uint32_t chunk_tokens,
                                             std::uint32_t frontier, std::uint32_t fallback,
                                             bool query_protected) noexcept {
    try {
        if (!chunk_tokens || sequence.text_kv_valid != frontier) return;
        // An anchor/coverage image captured at this same completed chunk can
        // serve both roles. Equal token position alone never proves validity.
        if (reuse_memory_input_checkpoint(sequence, frontier, fallback, query_protected)) return;
        copy_tail(sequence, prefill_hidden.slice(1, static_cast<std::int32_t>(chunk_tokens) - 1, 1));
        auto captured = capture_memory_checkpoint(sequence, true);
        sequence.window.checkpoints.slot(kvmem::CheckpointPurpose::InputReuse).data = std::move(captured);
        if (kvmem_options.verify_transfers)
            std::fprintf(stderr, "KVMEM_INPUT_REUSE action=capture frontier=%u earlier=%u extra_tokens=0 limit=%llu fallback=%u query_protected=%d\n",
                frontier, frontier, static_cast<unsigned long long>(sequence.window.coverage.policy().extra_recompute_tokens),
                fallback, int(query_protected));
    } catch (const std::exception& error) {
        // Optional input reuse must not consume the prior safe rewrite source,
        // an in-flight Query source, or the previously retained input image.
        std::fprintf(stderr, "KVMEM_INPUT_REUSE action=failed frontier=%u fallback=%u reason=%s\n",
            frontier, fallback, error.what());
    }
}

void ProgramImpl::prune_completed_memory_checkpoints(SequenceState& sequence) {
    auto& window = sequence.window;
    const auto limit = window.coverage.policy().extra_recompute_tokens;
    if (!limit) return;
    std::vector<std::shared_ptr<KvmemPrefixCheckpoint>> checkpoints;
    std::vector<CheckpointRecomputePoint> points;
    const auto* endpoint = memory_checkpoint(window, KvmemCheckpointKind::Endpoint);
    for (const auto kind : checkpoint_kinds(window)) {
        const auto& cp = memory_checkpoint_slot(window, kind);
        if (!cp || std::find(checkpoints.begin(), checkpoints.end(), cp) != checkpoints.end()) continue;
        checkpoints.push_back(cp);
        const auto frontier = cp->prefix.size();
        // The newest endpoint serves append; an input-side representative serves
        // repeat/regenerate. Do not replace that latter role by a pre-Query point
        // or by an endpoint beyond the next prompt's legal restore limit.
        const auto minimum = window.query_end && frontier >= window.query_end &&
            frontier < window.output_begin ? window.query_end : 0U;
        points.push_back({frontier, minimum, complete_memory_checkpoint(sequence, *cp),
                          cp.get() == endpoint || frontier == sequence.execution_frontier});
    }
    const auto replacements = checkpoint_recompute_plan(points, limit);
    for (std::size_t i = 0; i < checkpoints.size(); ++i) {
        if (replacements[i] == no_checkpoint_surrogate) continue;
        const auto& cp = checkpoints[i];
        const auto& earlier = checkpoints[replacements[i]];
        const bool query_checkpoint = memory_checkpoint(window, KvmemCheckpointKind::Base) == cp.get();
        if (kvmem_options.verify_transfers)
            std::fprintf(stderr, "KVMEM_CHECKPOINT_PRUNE frontier=%zu earlier=%zu extra_tokens=%zu limit=%llu extra_exact_capture_required=%d source_cold=%d\n",
                cp->prefix.size(), earlier->prefix.size(), cp->prefix.size() - earlier->prefix.size(),
                static_cast<unsigned long long>(limit), int(query_checkpoint), int(!earlier->state));
        window.coverage.erase(cp.get());
        for (auto& slot : window.checkpoints) if (slot.data == cp) slot.data.reset();
        // Outstanding I/O and foreign owners retain their own shared references.
        // Removing this owner's aliases never releases their borrowed bytes.
    }
}

void ProgramImpl::capture_memory_prefix(SequenceState& sequence, std::uint32_t chunk_tokens, bool query_checkpoint) {
    auto& window = sequence.window;
    const auto* previous = memory_checkpoint(window, query_checkpoint ? KvmemCheckpointKind::Base : KvmemCheckpointKind::PromptTail);
    if (!chunk_tokens || (previous && previous->prefix.size() >= sequence.text_kv_valid)) {
        throw std::logic_error("KVMem prefix checkpoint is not at an uncaptured evaluated boundary");
    }
    copy_tail(sequence, prefill_hidden.slice(1, static_cast<std::int32_t>(chunk_tokens) - 1, 1));
    publish_memory_checkpoint_source(window, capture_memory_checkpoint(sequence), query_checkpoint);
    if (query_checkpoint) window.query_probe_swaps = window.swaps;
    if (kvmem_options.verify_transfers)
        std::fprintf(stderr, "KVMEM_CHECKPOINT_CAPTURE purpose=%s frontier=%u\n",
            query_checkpoint ? "query_replay" : "prompt_tail", sequence.text_kv_valid);
}

std::unique_ptr<KvmemPrefixCheckpoint> ProgramImpl::capture_memory_checkpoint(SequenceState& sequence, bool optional) {
    auto& window = sequence.window;
    const auto frontier = sequence.text_kv_valid;
    if (!frontier || frontier > sequence.ledger.size() || frontier != window.statistics_frontier ||
        window.graph.accepted_through!=frontier ||
        !sequence.tail_hidden_valid || sequence.state.fork_pending ||
        (backend_kv_pages && sequence.mtp_kv_valid != frontier)) {
        throw std::logic_error("KVMem checkpoint is not a complete committed boundary");
    }
    WindowMemoryBackend backend(*this, sequence, frontier);
    auto descriptor = backend.descriptor();
    kvmem::CheckpointIdentity identity{std::move(descriptor.identity), std::move(descriptor.layout), backend.stamp()};
    const auto history = window.selector->retrieval_checkpoint();
    const auto& statistics = window.statistics->config();
    const auto mean_bytes = frontier % statistics.block_tokens
        ? size_t(statistics.n_layer) * (size_t(statistics.n_embd_k) + 1) * sizeof(float) : 0;
    const auto metadata = sizeof(KvmemPrefixCheckpoint) + KvmemPrefixCheckpoint::identity_bytes(identity) +
        size_t(frontier)*sizeof(TokenId) + sequence.prefix_identity.allocated_bytes() +
        window.pages.size()*sizeof(uint64_t) + mean_bytes + history.scores->capacity()*sizeof(double) +
        window.graph.bytes() + window.feedback.snapshot().bytes();
    const auto bytes = state_images->host_layout().image_bytes;
    // Copying then truncating identity retains its full allocation. Admit that
    // capacity, all attachments and the native image before constructing them.
    checkpoint_memory_room(sequence.window,(optional ? 2*bytes : bytes)+metadata,optional);
    auto checkpoint = std::make_unique<KvmemPrefixCheckpoint>();
    checkpoint->identity = std::move(identity);
    checkpoint->prefix.assign(sequence.ledger.begin(), sequence.ledger.begin() + frontier);
    checkpoint->prefix_identity = sequence.prefix_identity;
    checkpoint->prefix_identity.truncate(frontier);
    checkpoint->prefix_digest = sequence.prefix_digests.at(frontier);
    checkpoint->mean_tail = window.statistics->mean_checkpoint(frontier);
    checkpoint->history = history;
    checkpoint->graph=window.graph;
    checkpoint->feedback=window.feedback.snapshot();
    checkpoint->attention_scope=window.attention_scope;
    checkpoint->resident_pages.reserve(window.pages.size());
    const auto pages = kv_pages_for_tokens(frontier);
    for (auto id : window.pages) if (id < pages) checkpoint->resident_pages.push_back(id);
    if (checkpoint->metadata_bytes() > metadata)
        throw std::logic_error("checkpoint metadata exceeded admission");
    checkpoint->state = memory_checkpoint_pool->acquire(bytes);
    std::memset(checkpoint->state->data(), 0, bytes);
    archive_memory_window(sequence);
    std::array<std::byte*, 1> host{static_cast<std::byte*>(checkpoint->state->data())};
    state_images->copy_to_host_segments(state_store->physical_slot(sequence.state.write),
        host, bytes, compute_streams);
    device.synchronize();
    if (kvmem_options.verify_transfers) {
        std::fprintf(stderr, "KVMEM_PREFIX_CHECKPOINT frontier=%u state_bytes=%zu metadata_bytes=%llu admitted_metadata_bytes=%zu\n",
            frontier, bytes, (unsigned long long)checkpoint->metadata_bytes(), metadata);
        if (window.coverage.policy().enabled) {
            const auto staging = kvmem_options.disk_bytes ? bytes : 0;
            // The image is already in the pool; its unpublished metadata is not
            // yet reachable through the retained/active checkpoint directories.
            const auto charged = memory_summary().kvmem.checkpoint_bytes + checkpoint->metadata_bytes() + staging;
            std::fprintf(stderr, "KVMEM_CHECKPOINT_HOST charged_bytes=%llu budget_bytes=%llu\n",
                (unsigned long long)charged, (unsigned long long)(window.coverage.policy().host_bytes*max_concurrency));
        }
    }
    return checkpoint;
}

std::shared_ptr<KvmemPrefixCheckpoint>& ProgramImpl::memory_checkpoint_slot(KvmemWindowState& window,
                                                                          KvmemCheckpointKind kind) {
    switch (kind) {
    case KvmemCheckpointKind::Base: return window.checkpoints.slot(kvmem::CheckpointPurpose::QueryReplay).data;
    case KvmemCheckpointKind::Endpoint: return window.checkpoints.slot(kvmem::CheckpointPurpose::Endpoint).data;
    case KvmemCheckpointKind::InputFallback: return window.checkpoints.slot(kvmem::CheckpointPurpose::InputFallback).data;
    case KvmemCheckpointKind::Reconstruction: return window.checkpoints.slot(kvmem::CheckpointPurpose::Reconstruction).data;
    case KvmemCheckpointKind::ReuseAnchor: return window.checkpoints.slot(kvmem::CheckpointPurpose::ReuseAnchor).data;
    case KvmemCheckpointKind::PromptTail: return window.checkpoints.slot(kvmem::CheckpointPurpose::PromptTail).data;
    case KvmemCheckpointKind::InputReuse: return window.checkpoints.slot(kvmem::CheckpointPurpose::InputReuse).data;
    }
    const auto index = unsigned(kind)-unsigned(KvmemCheckpointKind::CoverageBegin);
    if (auto* entry=window.coverage.at(index)) return entry->record.data;
    throw std::logic_error("invalid KVMem checkpoint kind");
}

const KvmemPrefixCheckpoint* ProgramImpl::memory_checkpoint(const KvmemWindowState& window,
                                                           KvmemCheckpointKind kind) const {
    kvmem::CheckpointPurpose purpose;
    switch (kind) {
    case KvmemCheckpointKind::Base: purpose = kvmem::CheckpointPurpose::QueryReplay; break;
    case KvmemCheckpointKind::Endpoint: purpose = kvmem::CheckpointPurpose::Endpoint; break;
    case KvmemCheckpointKind::InputFallback: purpose = kvmem::CheckpointPurpose::InputFallback; break;
    case KvmemCheckpointKind::Reconstruction: purpose = kvmem::CheckpointPurpose::Reconstruction; break;
    case KvmemCheckpointKind::ReuseAnchor: purpose = kvmem::CheckpointPurpose::ReuseAnchor; break;
    case KvmemCheckpointKind::PromptTail: purpose = kvmem::CheckpointPurpose::PromptTail; break;
    case KvmemCheckpointKind::InputReuse: purpose = kvmem::CheckpointPurpose::InputReuse; break;
    default: {
        const auto* entry=window.coverage.at(unsigned(kind)-unsigned(KvmemCheckpointKind::CoverageBegin));
        return entry ? entry->record.data.get() : nullptr;
    }
    }
    const auto record = window.checkpoints.find(purpose);
    return record ? record->data.get() : nullptr;
}

std::uint32_t ProgramImpl::memory_observed_prefix(const PreparedPromptData& prompt) const {
    if (!prompt.identity.reusable) return 0;
    std::size_t best = 0;
    for (const auto& history : memory_histories) {
        if (!history || !history->stamp.valid || history->session_key != prompt.context_cache.session_key) continue;
        for (const auto kind : checkpoint_kinds(*history)) {
            const auto* checkpoint = memory_checkpoint(*history, kind);
            if (!checkpoint) continue;
            const auto limit = std::min(prompt.token_ids.size(), checkpoint->prefix.size());
            std::size_t common = 0;
            while (common < limit && prompt.token_ids[common] == checkpoint->prefix[common]) ++common;
            // This is a capture hint, never permission to restore old state.
            best = std::max(best, common);
        }
    }
    return static_cast<std::uint32_t>(best);
}

std::uint32_t ProgramImpl::memory_reuse_anchor(const PreparedPromptData& prompt, std::uint32_t base,
                                             std::uint32_t observed) const {
    const auto end = prompt.token_ids.size() > 0 ? prompt.token_ids.size() - 1 : 0;
    const auto workspace=backend_kv_pages ? std::max(draft_window,ngram_draft_window)+draft_window : 0;
    const auto query = prompt.memory_query && prompt.token_ids.size()+workspace > kvmem_window_tokens
        ? std::max<std::size_t>(base, memory_query_span(prompt).begin) : std::max<std::size_t>(base, end);
    std::vector<kvmem::PrefixRange> groups;
    for (const auto& item : prompt.vision_items) {
        if (!item.token_spans.empty()) {
            const auto& first = item.token_spans.front();
            const auto& last = item.token_spans.back();
            groups.push_back({first.begin, last.begin + last.count});
        }
    }
    return static_cast<std::uint32_t>(kvmem::prefix_anchor(base, observed, query, groups));
}

void ProgramImpl::capture_memory_anchor(SequenceState& sequence, std::uint32_t chunk_tokens) noexcept {
    try {
        auto& window = sequence.window;
        for (const auto kind : checkpoint_kinds(window)) {
            const auto& candidate = memory_checkpoint_slot(window, kind);
            if (candidate && candidate->prefix.size() == sequence.text_kv_valid) {
                auto shared = candidate;
                memory_checkpoint_slot(window, KvmemCheckpointKind::ReuseAnchor) = std::move(shared);
                window.coverage.remember({kvmem::CheckpointPurpose::ReuseAnchor,
                    memory_checkpoint_slot(window,KvmemCheckpointKind::ReuseAnchor)},kvmem::CoveragePurpose::Anchor);
                window.checkpoints.prune();
                progress_checkpoint_coverage(sequence.window);
                return;
            }
        }
        if (!chunk_tokens) return; // A restored boundary can only alias an existing complete state.
        if (skip_memory_coverage(sequence)) return;
        copy_tail(sequence, prefill_hidden.slice(1, static_cast<std::int32_t>(chunk_tokens) - 1, 1));
        memory_checkpoint_slot(window, KvmemCheckpointKind::ReuseAnchor) = capture_memory_checkpoint(sequence);
        window.coverage.remember({kvmem::CheckpointPurpose::ReuseAnchor,
            memory_checkpoint_slot(window,KvmemCheckpointKind::ReuseAnchor)},kvmem::CoveragePurpose::Anchor);
        window.checkpoints.prune();
        progress_checkpoint_coverage(sequence.window);
        if (kvmem_options.verify_transfers)
            std::fprintf(stderr, "KVMEM_REUSE_ANCHOR frontier=%u committed=1\n", sequence.text_kv_valid);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "KVMEM_REUSE_ANCHOR capture skipped: %s\n", error.what());
    }
}

void ProgramImpl::capture_memory_endpoint(SequenceState& sequence, const RequestControl& request) noexcept {
    if (!kvmem_window_tokens || !request.allow_memory_reuse || !sequence.window.statistics ||
        request.lifecycle != Lifecycle::Finishable) return;
    try {
        sequence.window.graph.reserve_extra(sequence.window.feedback.snapshot().bytes());
        sequence.window.feedback.finish_output();
        sequence.window.graph.include_feedback(sequence.window.feedback.snapshot().bytes());
    }
    catch(const std::exception& error) {
        std::fprintf(stderr,"KVMEM_ATTENTION finish failed: %s\n",error.what()); return;
    }
    const auto frontier = sequence.execution_frontier;
    if (sequence.text_kv_valid != frontier || !sequence.tail_hidden_valid ||
        sequence.state.fork_pending || (backend_kv_pages && sequence.mtp_kv_valid != frontier)) return;
    if (!sequence.window.coverage.policy().extra_recompute_tokens &&
        !sequence.window.query_replay_active &&
        memory_checkpoint(sequence.window, KvmemCheckpointKind::PromptTail)) {
        // Normal finish has accepted the replay. Retire its superseded T=0
        // execution role before allocating the endpoint, preserving the original
        // single-Base peak rather than holding Query + tail + a new endpoint.
        memory_checkpoint_slot(sequence.window, KvmemCheckpointKind::Base).reset();
    }
    for (const auto kind : kKvmemCheckpointKinds)
        if (const auto* checkpoint = memory_checkpoint(sequence.window, kind);
            checkpoint && checkpoint->prefix.size() == frontier) return;
    try {
        const auto started = Clock::now();
        sequence.window.checkpoints.slot(kvmem::CheckpointPurpose::Endpoint).data = capture_memory_checkpoint(sequence);
        sequence.window.coverage.remember(sequence.window.checkpoints.slot(kvmem::CheckpointPurpose::Endpoint),kvmem::CoveragePurpose::Endpoint);
        progress_checkpoint_coverage(sequence.window);
        if (kvmem_options.verify_transfers)
            std::fprintf(stderr, "KVMEM_ENDPOINT frontier=%u committed=1 capture_ms=%.3f\n", frontier,
                std::chrono::duration<double, std::milli>(Clock::now() - started).count());
    } catch (const std::exception& error) {
        std::fprintf(stderr, "KVMEM_ENDPOINT capture skipped: %s\n", error.what());
    }
}

bool ProgramImpl::memory_user_already_evaluated(const PreparedPromptData& prompt,
                                               std::uint32_t frontier) const {
    if (!prompt.memory_query) return false;
    const auto query = memory_query_span(prompt);
    // Caller proved the complete token/position prefix. Continuation is an
    // input-boundary fact and does not require retained Query feature rows.
    return query.count && query.begin+query.count<=frontier;
}

void ProgramImpl::capture_memory_rewrite(SequenceState& sequence, KvmemCheckpointKind kind) noexcept {
    auto& window = sequence.window;
    const auto* purpose = kind == KvmemCheckpointKind::InputFallback ? "input" : "reconstruction";
    try {
        if (kind != KvmemCheckpointKind::InputFallback && kind != KvmemCheckpointKind::Reconstruction)
            throw std::logic_error("invalid KVMem rewrite checkpoint purpose");
        if ((!window.checkpoints.slot(kvmem::CheckpointPurpose::QueryReplay).data ||
             window.checkpoints.slot(kvmem::CheckpointPurpose::QueryReplay).data->prefix.size() != sequence.text_kv_valid) &&
            (!window.checkpoints.slot(kvmem::CheckpointPurpose::PromptTail).data ||
             window.checkpoints.slot(kvmem::CheckpointPurpose::PromptTail).data->prefix.size() != sequence.text_kv_valid))
            memory_checkpoint_slot(window, kind) = capture_memory_checkpoint(sequence);
        if (kvmem_options.verify_transfers)
            std::fprintf(stderr, "KVMEM_REWRITE purpose=%s frontier=%u canonical=1\n",
                         purpose, sequence.text_kv_valid);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "KVMEM_REWRITE purpose=%s capture skipped: %s\n", purpose, error.what());
    }
    if (kind == KvmemCheckpointKind::Reconstruction) window.reconstruction_frontier = 0;
}

void ProgramImpl::capture_memory_reconstruction_if_ready(SequenceState& sequence) {
    auto& window = sequence.window;
    if (window.reconstruction_frontier < sequence.text_kv_valid) window.reconstruction_frontier = 0;
    if (!kvmem_window_tokens || !requests[sequence.lane].allow_memory_reuse ||
        !window.reconstruction_frontier || window.reconstruction_frontier != sequence.text_kv_valid ||
        sequence.execution_frontier != sequence.text_kv_valid ||
        (backend_kv_pages && sequence.mtp_kv_valid != sequence.text_kv_valid)) return;
    capture_memory_rewrite(sequence, KvmemCheckpointKind::Reconstruction);
}

bool ProgramImpl::rewind_memory_query(SequenceState& sequence) {
    auto& window = sequence.window;
    if (!window.checkpoints.slot(kvmem::CheckpointPurpose::QueryReplay).data || window.query_replayed ||
        window.query_tokens != window.query_end - window.query_begin ||
        sequence.state.read != sequence.state.write || sequence.state.fork_pending) {
        throw std::logic_error("KVMem query replay has no complete exclusive checkpoint");
    }
    const auto& checkpoint = *window.checkpoints.slot(kvmem::CheckpointPurpose::QueryReplay).data;
    if (checkpoint.prefix.size() != window.query_begin) {
        throw std::logic_error("KVMem query checkpoint is not the current query boundary");
    }
    const auto frontier=uint32_t(checkpoint.prefix.size());
    const auto pages=kv_pages_for_tokens(frontier);
    window.query_required=kvmem::query_prefix_required(frontier,page_tokens,
        kvmem_options.sink_blocks(),kvmem_options.recent_blocks(),window.media_pages);
    score_memory_query(sequence,window.query_required,&checkpoint);
    std::vector<uint32_t> candidates;
    std::vector<double> scores;
    for(uint32_t id=0;id<pages;++id) {candidates.push_back(id);scores.push_back(window.selector->selection_score(id));}
    window.query_selected=window.selector->pick_scored_blocks(candidates,scores,window.query_required,
        kvmem_options.selected_tokens/page_tokens,true);
#ifdef NINFER_KVMEM_QUERY_TEST_HOOKS
    std::fprintf(stderr,"KVMEM_QUERY_TEST phase=selected lane=%u rows=%u pages=",sequence.lane,window.query_tokens);
    for (auto page:window.query_selected) std::fprintf(stderr,"%u,",page);
    std::fprintf(stderr,"\n");
#endif
    const auto admission=kvmem::query_replay_plan({window.query_begin,window.query_end},frontier,
        sequence.text_kv_valid,kvmem_options.query_replay_max_tokens,page_tokens,
        window.query_selected.size(),kvmem_window_tokens/page_tokens,
        backend_kv_pages ? std::max(draft_window,ngram_draft_window)+draft_window : 0);
    auto next=window.query_selected;
    for(auto id=frontier/page_tokens;id<kv_pages_for_tokens(sequence.text_kv_valid);++id)
        next.push_back(id);
    std::sort(next.begin(),next.end());
    next.erase(std::unique(next.begin(),next.end()),next.end());
    const bool unchanged=window.query_probe_swaps==window.swaps && next==window.pages;
    if(!admission.admitted || unchanged) {
        window.query_replayed=true;
        window.query_selected.clear();window.query_required.clear();
        if(kvmem_options.verify_transfers) std::fprintf(stderr,
            "KVMEM_PREFILL_DECISION path=%s reason=%s replay_rows=0\n",
            unchanged?"unchanged_selection":"query_replay_skipped",
            unchanged?"same_attention_view":admission.reason);
        return false;
    }
    const auto previous_frontier = sequence.text_kv_valid;
    window.query_replayed = true; // Probe Q stays frozen while replay replaces K once.
    window.query_replay_active=true;
    restore_memory_prefix(sequence, checkpoint, false, true);
    if (kvmem_options.verify_transfers) {
        std::fprintf(stderr, "KVMEM_QUERY_REPLAY from=%u to=%u query_tokens=%u replay_rows=%u prefix_match=1 state_exact=1\n",
            previous_frontier, sequence.text_kv_valid, window.query_tokens,previous_frontier-sequence.text_kv_valid);
    }
    return true;
}

void ProgramImpl::restore_memory_prefix(SequenceState& sequence, const KvmemPrefixCheckpoint& checkpoint,
                                        bool preserve_view, bool allow_pending, bool resident_native) {
    auto& window = sequence.window;
    const auto frontier = static_cast<std::uint32_t>(checkpoint.prefix.size());
    WindowMemoryBackend backend(*this, sequence, sequence.text_kv_valid);
    const auto descriptor = backend.descriptor();
    const kvmem::CheckpointIdentity expected{descriptor.identity, descriptor.layout, checkpoint.identity.stamp};
    if (!checkpoint.identity.compatible_with(expected) ||
        !(checkpoint.identity.stamp.session == window.stamp.session) ||
        frontier != checkpoint.identity.stamp.frontier || frontier > sequence.ledger.size() ||
        !std::equal(checkpoint.prefix.begin(), checkpoint.prefix.end(), sequence.ledger.begin()) ||
        !checkpoint.prefix_identity.prefix_equals(sequence.prefix_identity, frontier)) {
        throw std::logic_error("KVMem rejected a stale query checkpoint before mutation");
    }
    const auto bytes = state_images->host_layout().image_bytes;
    // All prefix records were made durable at capture. Query appends cannot alter
    // native values/scales in earlier rows; the partial page is explicitly clipped.
    const auto pages = kv_pages_for_tokens(frontier);
    if (window.archive.size() < pages) throw std::logic_error("KVMem checkpoint lost Host prefix");
    for (std::uint32_t id = 0; id < pages; ++id) {
        if (!window.archive[id] || window.host_versions[id] < valid_columns(id, frontier)) {
            throw std::logic_error("KVMem checkpoint has an incomplete native prefix");
        }
    }
    if(checkpoint.graph.accepted_through!=frontier || checkpoint.graph.block_tokens!=page_tokens)
        throw std::logic_error("KVMem checkpoint graph has a different accepted prefix");
    if (!checkpoint.state) {
        checkpoint_memory_room(sequence.window, bytes);
        auto loaded = memory_checkpoint_pool->acquire(bytes);
        checkpoint.read_state(loaded->data(), loaded->size());
        auto& mutable_checkpoint = const_cast<KvmemPrefixCheckpoint&>(checkpoint);
        mutable_checkpoint.state = std::move(loaded);
        mutable_checkpoint.cold_state = {};
    }
    auto restored_graph=checkpoint.graph;
    restored_graph.set_messages(window.request_messages);
    auto restored_feedback=window.feedback;
    restored_feedback.restore(checkpoint.feedback);
    window.statistics->truncate_to(frontier);
    window.statistics->restore_mean_checkpoint(frontier, checkpoint.mean_tail);
    window.selector->truncate_to(frontier);
    if (preserve_view) window.selector->restore_retrieval(checkpoint.history);
    window.graph=std::move(restored_graph);
    ++window.graph.epoch;
    window.feedback=std::move(restored_feedback);
    // Query replay happens inside this request's prefill: preserve its fresh
    // generation parser state. A resumed native continuation restores its own.
    if(preserve_view) window.attention_scope=checkpoint.attention_scope;
    window.statistics_frontier = frontier;
    window.archive.resize(pages);
    window.host_versions.resize(pages);
    window.payload_bytes = pages * backend.page_bytes;
    if (frontier % page_tokens) window.host_versions.back() = frontier % page_tokens;
    // A consumed history has one archive. Rewinding explicitly revokes later states
    // before their payload/statistics can be overwritten; no full-history COW is needed.
    window.coverage.truncate(frontier);
    for (const auto kind : kKvmemCheckpointKinds) {
        auto& slot = memory_checkpoint_slot(window, kind);
        if (slot && slot->prefix.size() > frontier) slot.reset();
    }
    sequence.text_kv_valid = frontier;
    if (speculative_backend == SpeculativeBackend::DFlash2) {
        sequence.dflash_context_frontier = frontier;
        const auto recent = static_cast<std::uint64_t>(kvmem_options.recent_blocks()) * page_tokens;
        sequence.dflash_context_floor = frontier > recent
            ? static_cast<std::uint32_t>(frontier - recent) : 0U;
    }
    if (backend_kv_pages) { sequence.mtp_kv_valid = frontier; sequence.mtp_draft_count = 0; }
    ++window.stamp.execution_history;
    state_store->begin_active_overwrite(sequence.state.write);
    std::array<const std::byte*, 1> host{static_cast<const std::byte*>(checkpoint.state->data())};
    state_images->copy_from_host_segments(host, bytes,
        state_store->physical_slot(sequence.state.write), compute_streams);
    device.synchronize();
    if (kvmem_options.verify_transfers) {
        PinnedHostBuffer verify(bytes);
        std::memset(verify.data(), 0, bytes);
        std::array<std::byte*, 1> destination{static_cast<std::byte*>(verify.data())};
        state_images->copy_to_host_segments(state_store->physical_slot(sequence.state.write),
            destination, bytes, compute_streams);
        device.synchronize();
        if (std::memcmp(verify.data(), checkpoint.state->data(), bytes)) {
            throw std::logic_error("KVMem query StateImage restore changed native bytes");
        }
    }
    if (preserve_view || !window.query_tokens || window.query_tokens!=window.query_end-window.query_begin) {
        auto restored = std::make_shared<WindowMemoryBackend>(*this, sequence, frontier, resident_native);
        kvmem::MemorySession session(restored);
        // A checkpoint includes already appended reserve rows. Restore its exact
        // view within B+R, then ordinary append pressure restores the B/R split.
        session.prepare(restored->stamp(), checkpoint.resident_pages, checkpoint.resident_pages,
            {kvmem_window_tokens, 0, kvmem_window_tokens, memory_retained_budget});
        for (;;) {
            const auto epoch = memory_io ? memory_io->completion_epoch() : 0;
            if (session.advance() == kvmem::MemoryPhase::Ready) break;
            if (memory_io && kvmem_options.disk_bytes && session.phase() == kvmem::MemoryPhase::Spilling)
                memory_io->wait_for_completion(epoch, std::chrono::milliseconds(1));
        }
        session.publish();
    } else {
        prepare_window(sequence, frontier, frontier, true, allow_pending,kvmem::SelectionReason::Query);
    }
    sequence.tail_hidden_valid = true;
}

KvmemHistoryPreparation::~KvmemHistoryPreparation() { ticket.cancel(); }

std::optional<KvmemHistoryMatch> ProgramImpl::memory_restorable_history(
    const PreparedPromptData& prompt, const PrefixShortlistDigests* digests) const {
    if (!kvmem_window_tokens || !prompt.identity.reusable || (prompt.context_cache.session_key &&
        !prompt.context_cache.update_session_index)) return std::nullopt;
    const auto started = Clock::now();
    const auto limit = kvmem::prefix_restore_limit(prompt.token_ids.size(), prompt.token_ids.size(),
                                                   prompt.token_ids.size(), true);
    struct Candidate {
        KvmemHistoryMatch match;
        const KvmemPrefixCheckpoint* checkpoint;
        uint64_t cold_bytes;
        size_t rank;
    };
    unsigned prefix_rejected=0, identity_rejected=0, query_rejected=0, incomplete=0, branch_rejected=0;
    unsigned shortlisted=0, exact_checks=0;
    uint64_t compared_tokens=0;
    PrefixShortlistDigests incoming;
    std::optional<KvmemHistoryMatch> best;
    bool fork=false;
    for (unsigned pass=0; pass<2 && !best; ++pass) {
        // Tool continuation remains append plus resident-only Pressure. Only a
        // named request with a current user Query may discover another owner.
        if (pass && (!prompt.context_cache.session_key || !prompt.memory_query ||
                     !prompt.memory_query->count)) break;
        std::vector<Candidate> candidates;
        for (size_t index=0; index<memory_histories.size(); ++index) {
            const auto& history=memory_histories[index];
            if (!history || !history->stamp.valid) continue;
            const bool own=history->session_key==prompt.context_cache.session_key;
            if (own != (pass==0)) continue;
            if (pass && (!history->session_key || !history->coverage.policy().enabled)) continue;
            if (!pass && history->coverage.policy().enabled && !history->session_key) {
                const KvmemPrefixCheckpoint* latest=nullptr;
                for (const auto kind:checkpoint_kinds(*history)) {
                    const auto* cp=memory_checkpoint(*history,kind);
                    if(cp && (!latest || cp->prefix.size()>latest->prefix.size())) latest=cp;
                }
                if(!latest) continue;
                const auto shared=std::mismatch(latest->prefix.begin(),latest->prefix.end(),
                    prompt.token_ids.begin(),prompt.token_ids.end()).first-latest->prefix.begin();
                if(!kvmem::checkpoint_anonymous_continuation(shared,history->output_begin,latest->prefix.size()) ||
                   !latest->prefix_identity.matches(prompt,std::min<size_t>(shared,history->output_begin))) {
                    ++branch_rejected; continue;
                }
            }
            std::unordered_set<const KvmemPrefixCheckpoint*> seen;
            size_t rank=0;
            for(const auto kind:checkpoint_kinds(*history)) {
                const auto* cp=memory_checkpoint(*history,kind);
                const auto order=rank++;
                if(!cp || !seen.insert(cp).second || !cp->prefix.size() || cp->prefix.size()>limit) continue;
                if(pass) {
                    if(!digests || digests->size()!=prompt.token_ids.size()) {
                        incoming.assign(prompt); digests=&incoming;
                    }
                    if(cp->prefix_digest!=digests->at(cp->prefix.size())) { ++prefix_rejected; continue; }
                }
                uint64_t cold=cp->state ? 0 : cp->state_bytes();
                for(const auto page:cp->resident_pages) {
                    if(page>=history->archive.size()) { cold=UINT64_MAX; break; }
                    const auto& record=history->archive[page];
                    if(!record.resident_bytes()) cold+=record.size();
                }
                candidates.push_back({{index,kind,static_cast<uint32_t>(cp->prefix.size())},cp,cold,order});
            }
        }
        std::stable_sort(candidates.begin(),candidates.end(),[&](const auto& a,const auto& b) {
            if(a.match.frontier!=b.match.frontier) return a.match.frontier>b.match.frontier;
            if(pass && a.cold_bytes!=b.cold_bytes) return a.cold_bytes<b.cold_bytes;
            if(a.match.history!=b.match.history)
                return memory_histories[a.match.history]->publication_order>memory_histories[b.match.history]->publication_order;
            return a.rank<b.rank;
        });
        shortlisted+=static_cast<unsigned>(candidates.size());
        unsigned foreign_checks=0;
        for(const auto& candidate:candidates) {
            if(pass && foreign_checks++==16) break;
            const auto& checkpoint=*candidate.checkpoint;
            const auto& history=*memory_histories[candidate.match.history];
            const auto frontier=candidate.match.frontier;
            const auto pages=kv_pages_for_tokens(frontier);
            bool complete=checkpoint.state_bytes()==state_images->host_layout().image_bytes &&
                frontier==checkpoint.identity.stamp.frontier && history.statistics && history.selector &&
                checkpoint.identity.stamp.session==history.stamp.session &&
                history.statistics_frontier>=frontier && history.archive.size()>=pages && history.host_versions.size()>=pages;
            for(uint32_t page=0;complete && page<pages;++page)
                complete=history.archive[page] && history.host_versions[page]>=valid_columns(page,frontier);
            if(!complete) { ++incomplete; continue; }
            ++exact_checks; compared_tokens+=frontier;
            if(frontier>=prompt.token_ids.size() ||
               !std::equal(checkpoint.prefix.begin(),checkpoint.prefix.end(),prompt.token_ids.begin())) {
                ++prefix_rejected; continue;
            }
            if(!checkpoint.prefix_identity.matches(prompt,frontier)) { ++identity_rejected; continue; }
            const auto query=memory_query_span(prompt);
            const auto workspace=backend_kv_pages ? std::max(draft_window,ngram_draft_window)+draft_window : 0;
            const auto admission=kvmem::query_replay_plan({query.begin,query.begin+query.count},query.begin,
                prompt.token_ids.size(),kvmem_options.query_replay_max_tokens,page_tokens,
                std::min<uint64_t>((query.begin+63)/64,kvmem_options.selected_tokens/page_tokens),
                kvmem_window_tokens/page_tokens,workspace);
            const bool needs_recall=checkpoint.resident_pages.size()<pages ||
                prompt.token_ids.size()+workspace>kvmem_window_tokens;
            if(admission.admitted && needs_recall && prompt.memory_query && frontier>query.begin &&
               !memory_user_already_evaluated(prompt,frontier)) { ++query_rejected; continue; }
            best=candidate.match; fork=pass!=0; break;
        }
    }
    if(kvmem_options.verify_transfers) std::fprintf(stderr,
        "KVMEM_HISTORY_LOOKUP frontier=%u prefix_rejected=%u identity_rejected=%u query_rejected=%u incomplete=%u branch_rejected=%u fork=%u candidates=%u exact_checks=%u compared_tokens=%llu lookup_us=%.3f\n",
        best ? best->frontier : 0,prefix_rejected,identity_rejected,query_rejected,incomplete,branch_rejected,
        unsigned(fork),shortlisted,exact_checks,(unsigned long long)compared_tokens,
        std::chrono::duration<double,std::micro>(Clock::now()-started).count());
    return best;
}

std::unique_ptr<KvmemWindowState> ProgramImpl::fork_memory_history(
    const KvmemWindowState& source, uint32_t frontier) {
    const auto started=Clock::now();
    const auto pages=kv_pages_for_tokens(frontier);
    std::unordered_map<const KvmemPrefixCheckpoint*,std::shared_ptr<KvmemPrefixCheckpoint>> copies;
    size_t metadata=0;
    for(const auto kind:checkpoint_kinds(source)) {
        const auto* cp=memory_checkpoint(source,kind);
        if(cp && cp->prefix.size()<=frontier && copies.emplace(cp,nullptr).second) metadata+=cp->metadata_bytes();
    }
    // Optional optimization: never evict the source to make room for its fork.
    checkpoint_memory_room(const_cast<KvmemWindowState&>(source),metadata,true);
    auto child=std::make_unique<KvmemWindowState>();
    child->stamp=source.stamp;
    child->stamp.session={next_window_session.fetch_add(1),1};
    child->stamp.frontier=frontier;
    child->stamp.history_revision=1;
    child->stamp.execution_history=1;
    child->statistics=source.statistics->fork_mean_prefix(frontier);
    child->statistics_frontier=frontier;
    child->selector=std::make_unique<kvmem::KvMemStore>(*source.selector);
    child->selector->truncate_to(frontier);
    child->graph=source.graph;
    child->feedback=source.feedback;
    child->request_messages=source.request_messages;
    child->query_rows.resize(source.query_rows.size());
    child->coverage=source.coverage;
    child->coverage.truncate(frontier);
    child->checkpoints=source.checkpoints;
    for(auto& [original,copy]:copies) {
        copy=std::make_shared<KvmemPrefixCheckpoint>(*original);
        copy->identity.stamp.session=child->stamp.session;
    }
    for(const auto kind:kKvmemCheckpointKinds) {
        auto& cp=memory_checkpoint_slot(*child,kind);
        if(cp) cp=cp->prefix.size()<=frontier ? copies.at(cp.get()) : nullptr;
    }
    for(auto& entry:child->coverage) if(entry.info.identity) {
        entry.record.data=copies.at(entry.record.data.get());
        entry.info.identity=entry.record.data.get();
        entry.info.bytes=entry.record.checkpoint_info().bytes;
    }
    child->archive.reserve(pages);
    child->host_versions.assign(source.host_versions.begin(),source.host_versions.begin()+pages);
    for(uint32_t page=0;page<pages;++page) child->archive.push_back(source.archive[page].fork());
    child->payload_bytes=pages ? pages*source.archive.front().size() : 0;
    if(kvmem_options.verify_transfers) std::fprintf(stderr,
        "KVMEM_HISTORY_FORK frontier=%u shared_kv_bytes=%llu metadata_bytes=%zu statistics_bytes=%zu fork_us=%.3f\n",
        frontier,(unsigned long long)child->payload_bytes,metadata,child->statistics->allocated_bytes(),
        std::chrono::duration<double,std::micro>(Clock::now()-started).count());
    return child;
}

bool ProgramImpl::prepare_memory_materialization(MaterializationTransaction& transaction) {
    auto& staged=*requests[transaction.destination.value].prefill;
    if(!staged.memory_restore_frontier || !staged.memory_restore_generation) return true;
    auto& pending=transaction.memory_history;
    try {
        if(!pending) {
            const auto match=memory_restorable_history(staged.prompt,&materialization_prefix_digests_);
            if(!match || match->frontier!=staged.memory_restore_frontier ||
               memory_histories[match->history]->stamp.session.id!=staged.memory_restore_generation) {
                staged.memory_restore_generation=0; return true;
            }
            const auto& source=*memory_histories[match->history];
            pending=std::make_unique<KvmemHistoryPreparation>();
            pending->history=match->history;
            pending->generation=source.stamp.session.id;
            pending->selected=match->kind;
            if(source.session_key!=staged.prompt.context_cache.session_key) {
                uint64_t required=transaction.plan->impl_->memory_payload_reservation;
                for(const auto& active:requests) required+=active.memory_payload_reservation;
                const auto* checkpoint=memory_checkpoint(source,match->kind);
                const auto full_pages=match->frontier/page_tokens;
                for(size_t page=0;page<source.archive.size();++page) {
                    const bool selected=std::find(checkpoint->resident_pages.begin(),checkpoint->resident_pages.end(),page)!=checkpoint->resident_pages.end();
                    const bool shared=page<full_pages && page!=(match->frontier-1)/page_tokens &&
                        (!selected || source.archive[page].resident_bytes());
                    if(!shared) required+=source.archive[page].size();
                }
                std::unordered_set<const void*> cold;
                for(const auto kind:checkpoint_kinds(source)) if(const auto* cp=memory_checkpoint(source,kind);
                    cp && cp->cold_state && cold.insert(cp->cold_state.storage_identity()).second)
                    required+=cp->cold_state.size();
                if(required>memory_retained_budget)
                    throw std::runtime_error("fork cannot retain its source within the payload budget");
                pending->fork=fork_memory_history(source,match->frontier);
                pending->fork->session_key=staged.prompt.context_cache.session_key;
            }
        }
        if(pending->native_ready) return true;
        auto* history=pending->fork.get();
        if(!history) {
            auto& source=memory_histories[pending->history];
            if(!source || source->stamp.session.id!=pending->generation) {
                pending.reset(); staged.memory_restore_generation=0; return true;
            }
            history=&*source;
        }
        if(!memory_io || !kvmem_options.disk_bytes) return true;
        if(pending->ticket && !pending->ticket.ready()) return false;
        if(pending->ticket) {
            pending->ticket.wait();
            auto& work=*pending->work;
            if(pending->checkpoint) {
                auto& checkpoint=memory_checkpoint_slot(*history,*pending->checkpoint);
                if(!checkpoint || checkpoint->state || checkpoint->state_bytes()!=work.state.size())
                    throw std::logic_error("prepared state image changed before publication");
                checkpoint_memory_room(*history,work.state.size());
                auto state=memory_checkpoint_pool->acquire(work.state.size());
                std::memcpy(state->data(),work.state.data(),work.state.size());
                if(kvmem_options.verify_transfers) std::fprintf(stderr,
                    "KVMEM_COVERAGE_TIER read_bytes=%zu frontier=%zu fork=%u\n",work.state.size(),checkpoint->prefix.size(),unsigned(bool(pending->fork)));
                checkpoint->state=std::move(state);
                checkpoint->cold_state={};
                pending->checkpoint.reset(); pending->work.reset(); pending->ticket={};
            } else {
                for(size_t i=0;i<work.ids.size();++i) if(work.records[i]) {
                    work.records[i].seal();
                    history->archive.at(work.ids[i])=std::move(work.records[i]);
                }
                pending->native_ready=true;
                return true;
            }
        }
        for(const auto kind:checkpoint_kinds(*history)) {
            if((pending->fork || history->coverage.policy().enabled) && kind!=pending->selected) continue;
            const auto* checkpoint=memory_checkpoint(*history,kind);
            if(!checkpoint || checkpoint->state || !checkpoint->cold_state.size()) continue;
            if(memory_state_staging.charged_bytes()) return false;
            pending->checkpoint=kind;
            pending->work=std::make_shared<KvmemHistoryPreparation::Work>();
            auto work=pending->work;
            work->state_source=checkpoint->cold_state;
            work->state_tag=checkpoint->cold_tag;
            work->state_payload_bytes=checkpoint->state_bytes();
            pending->ticket=memory_io->submit(checkpoint->state_bytes(),[work,storage=memory_state_staging]() mutable {
                work->state=storage.allocate(work->state_payload_bytes);
                if(work->state_tag) kvmem::read_checkpoint_image(work->state_source,work->state_tag,{{work->state.writable_data(),work->state.size()}});
                else work->state_source.read(0,work->state.writable_data(),work->state.size());
            });
            return false;
        }
        pending->work=std::make_shared<KvmemHistoryPreparation::Work>();
        auto work=pending->work;
        const auto& checkpoint=*memory_checkpoint(*history,pending->selected);
        for(const auto id:checkpoint.resident_pages) {
            work->ids.push_back(static_cast<uint32_t>(id));
            work->sources.push_back(history->archive.at(id).source());
        }
        work->records.resize(work->ids.size()); work->leases.resize(work->ids.size());
        const auto stride=text_host_kv_page_stride+backend_host_kv_page_stride;
        const auto tail=(staged.memory_restore_frontier-1)/page_tokens;
        pending->ticket=memory_io->submit(work->ids.size()*stride,[work,storage=memory_payload_storage,stride,tail]() mutable {
            for(size_t i=0;i<work->ids.size();++i) {
                auto source=work->sources[i].lease();
                if(source.data() && work->ids[i]!=tail) work->leases[i]=std::move(source);
                else {
                    auto& record=work->records[i];
                    record=storage.allocate(stride);
                    source.read(0,record.writable_data(),stride);
                    record.seal();
                    work->leases[i]=record.lease();
                }
            }
            work->sources.clear();
        });
        return false;
    } catch(const std::exception& error) {
        std::fprintf(stderr,"KVMEM_HISTORY preparation failed; replaying prefix: %s\n",error.what());
        pending.reset(); staged.memory_restore_generation=0;
        return true;
    }
}

bool ProgramImpl::restore_memory_history(SequenceState& sequence, RequestControl::Prefill& staged) {
    const auto started = Clock::now();
    auto* transaction=std::get_if<MaterializationTransaction>(&context_transaction_);
    auto* prepared=transaction ? transaction->memory_history.get() : nullptr;
    if(!staged.memory_restore_generation || !prepared || prepared->generation!=staged.memory_restore_generation) return false;
    const bool fork=bool(prepared->fork);
    const auto kind=prepared->selected;
    const auto frontier=std::optional<uint32_t>(staged.memory_restore_frontier);
    if(fork) {
        sequence.window=std::move(*prepared->fork);
    } else {
        auto& history=memory_histories[prepared->history];
        if(!history || history->stamp.session.id!=prepared->generation) return false;
        sequence.window=std::move(*history);
        history.reset();
    }
    auto& window = sequence.window;
    const bool same_query = memory_user_already_evaluated(staged.prompt, *frontier);
    auto checkpoint = memory_checkpoint_slot(window, kind);
    window.coverage.touch(checkpoint.get());
    // The previous Program address-space lease has been released. These IDs
    // must never be treated as resident in the new native address space.
    window.pages.clear();
    window.removed_tokens = 0;
    window.reconstruction_frontier = 0;
    window.query_replayed = same_query;
    window.query_capture = false;
    window.query_replay_active = false;
    window.query_selected.clear();window.query_required.clear();
    const auto query = memory_query_span(staged.prompt);
    window.query_begin = static_cast<std::uint32_t>(query.begin);
    window.query_end = static_cast<std::uint32_t>(query.begin + query.count);
    // Query features are request-local seeds. The verified input prefix alone
    // determines continuation; never retain features from a discarded suffix.
    window.query_tokens = 0;
    for(auto& rows:window.query_rows) rows.clear();
    window.query_sums.clear();
    // The durable archive already covers the quoted prefix. This is not a
    // native allocation frontier; force-selection creates only B resident rows.
    sequence.text_kv_valid = *frontier;
    if (backend_kv_pages) sequence.mtp_kv_valid = *frontier;
    auto* materialization = std::get_if<MaterializationTransaction>(&context_transaction_);
    const bool resident_native = materialization && materialization->memory_history &&
        materialization->memory_history->native_ready;
    restore_memory_prefix(sequence, *checkpoint, true, false, resident_native);
    if (resident_native) materialization->memory_history.reset();

    // A pre-Query checkpoint already owns an uncommitted round whose scores
    // are the preceding committed history. Reuse it when editing that Query.
    // Starting again would count the same discarded Query in the round number.
    const auto restored_history=window.selector->retrieval_checkpoint();
    if (!restored_history.round || restored_history.committed)
        window.selector->begin_retrieval_round();
    if (backend_kv_pages) {
        // An MTP row combines target hidden at i with the embedding of token i+1.
        // Prefix identity proves only [0,frontier), so its last MTP row must be
        // rebuilt using this request's first suffix token before it is trusted.
        const auto last = *frontier - 1U;
        sequence.mtp_kv_valid = last;
        trim_sequence_kv(sequence, *frontier, last);
        ensure_sequence_kv_mapped(sequence, *frontier, *frontier);
        const auto selectors = state_selectors(sequence);
        execution::PrefillContext bridge{
            {device, parameters, work, state_images->linear(0),
             replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
             proposal_head, stage_runtime.get(), rope_yarn, fast_prefill_kernel,
             mtp_attention_window},
            text_kv_view(sequence), mtp_kv_view(sequence), decoder->text_kv,
            decoder->mtp_cache(), nullptr, *frontier, nullptr, nullptr,
            selectors.source, selectors.destination, 0, 0};
        bridge.cache_position_shift = window.removed_tokens;
        set_device_i32(io.rope_delta, sequence.rope_delta +
            checked_i32(window.removed_tokens, "restored MTP cache shift"));
        auto token = io.mtp->target_input_ids.slice(0, 0, 1);
        set_device_i32(token, staged.prompt.token_ids[*frontier]);
        const auto tokens = staged.prompt.token_ids.size();
        const std::array<std::int32_t, 3> rope{
            staged.prompt.positions.at(last), staged.prompt.positions.at(tokens + last),
            staged.prompt.positions.at(2 * tokens + last)};
        mark_workspace_usage(workspace_plan.mtp_prefill);
        if (staged.vision) {
            execution::mtp_bridge_multimodal(bridge, staged.prompt, *staged.vision,
                {.previous_hidden = &sequence.tail_hidden,
                 .position = checked_i32(last, "restored MTP tail"), .rope_position = rope});
        } else {
            execution::mtp_bridge_and_propose(bridge, token, sequence.tail_hidden,
                checked_i32(last, "restored MTP tail"), rope, false);
        }
        device.synchronize();
        work.reset();
        sequence.mtp_kv_valid = *frontier;
        commit_sequence_kv(sequence, *frontier, *frontier);
        window.host_versions.at(last / page_tokens) = 0;
        // Same-boundary query replay can restore before another view switch.
        archive_memory_window(sequence);
    }
    window.query_probe_swaps=window.swaps;
    if (*frontier == memory_checkpoint_frontier(sequence, staged.prompt_tokens, true) &&
        (memory_query_probe(sequence, staged.prompt_tokens) || !memory_input_reuse_frontier(staged.prompt)))
        publish_memory_checkpoint_source(window, std::move(checkpoint), memory_query_probe(sequence, staged.prompt_tokens));
    else
        // An exact replay can start at the input capture boundary, so prefill
        // will not cross it again. Keep the restored complete state for repeats
        // and cancellation recovery; restoring does not consume its payload.
        memory_checkpoint_slot(window, kind) = std::move(checkpoint);
    staged.base = *frontier;
    staged.cursor = *frontier;
    ++memory_history_hits;
    if (kvmem_options.verify_transfers) {
        std::fprintf(stderr, "KVMEM_HISTORY_RESTORE frontier=%u kind=%u same_query=%u prefix_match=1 state_exact=1 restore_ms=%.3f fork=%u\n",
            *frontier, static_cast<unsigned>(kind), same_query ? 1U : 0U,
            std::chrono::duration<double, std::milli>(Clock::now() - started).count(),unsigned(fork));
    }
    return true;
}

void ProgramImpl::reserve_memory_history(SequenceState& sequence, RequestControl& request,
                                         const PreparedPromptData& prompt) {
    const auto& key = prompt.context_cache.session_key;
    if (key && prompt.context_cache.update_session_index) {
        MemoryPublication* target = nullptr;
        for (auto& publication : memory_publications) {
            if (publication.key == key) { target = &publication; break; }
            if (!publication.key) { target = &publication; }
        }
        if (!target) {
            for (auto& publication : memory_publications) {
                const bool active = std::any_of(requests.begin(), requests.end(), [&](const auto& value) {
                    return value.memory_session_key == publication.key;
                });
                const bool retained = std::any_of(memory_histories.begin(), memory_histories.end(), [&](const auto& value) {
                    return value && value->session_key == publication.key;
                });
                if (!active && !retained && (!target || publication.order < target->order)) target = &publication;
            }
        }
        if (!target) throw std::logic_error("KVMem publication table has no reclaimable entry");
        if (target->key != key) { target->key = key; target->order = 0; }
        target->order = std::max(target->order, sequence.window.publication_order);
    }
    if (!sequence.window.statistics) ++memory_history_misses;
    // An explicit fresh request invalidates only its own lineage. A caller that
    // forbids index updates cannot consume or erase the named cached history.
    if (!request.allow_memory_reuse && prompt.context_cache.update_session_index) {
        for (auto& history : memory_histories) {
            if (history && history->session_key == prompt.context_cache.session_key &&
                history->publication_order <= sequence.window.publication_order) history.reset();
        }
    }
    std::uint64_t reserved = 0;
    for (const auto& active : requests) reserved += active.memory_payload_reservation;
    if (reserved > memory_retained_budget) {
        throw std::logic_error("KVMem active payload reservations exceed admission capacity");
    }
    auto available = memory_retained_budget - reserved;
    for (;;) {
        std::uint64_t cached_bytes = memory_idle_transfer ? memory_idle_transfer->work->destination.size() : 0;
        std::optional<std::size_t> victim;
        std::unordered_set<const void*> payloads,cold;
        // Active reservations already include the shared physical prefix.
        for(const auto& lane:continuation_states) if(lane.window.statistics)
            for(const auto& record:lane.window.archive) if(record) payloads.insert(record.storage_identity());
        const auto protected_source=request.prefill ? request.prefill->memory_restore_generation : 0;
        for (std::size_t index = 0; index < memory_histories.size(); ++index) {
            const auto& history = memory_histories[index];
            if (!history) continue;
            for(const auto& record:history->archive)
                if(record && payloads.insert(record.storage_identity()).second) cached_bytes+=record.size();
            std::vector<const KvmemPrefixCheckpoint*> seen;
            for (const auto kind : checkpoint_kinds(*history))
                if (const auto* checkpoint = memory_checkpoint(*history, kind);
                    checkpoint && std::find(seen.begin(), seen.end(), checkpoint) == seen.end()) {
                    if(checkpoint->cold_state && cold.insert(checkpoint->cold_state.storage_identity()).second)
                        cached_bytes += checkpoint->cold_state.size();
                    seen.push_back(checkpoint);
                }
            if(history->stamp.session.id!=protected_source &&
               (!victim || history->publication_order < memory_histories[*victim]->publication_order)) victim = index;
        }
        if (cached_bytes <= available) break;
        if (!victim) throw std::logic_error("KVMem retained Host accounting has no victim");
        memory_histories[*victim].reset();
        ++memory_history_evictions;
    }
}

void ProgramImpl::retain_memory_history(SequenceState& sequence, const RequestControl& request,
                                       bool completed) noexcept {
    // A cancelled request can retain an earlier complete checkpoint, but cannot
    // move a transaction whose backend still addresses the active sequence.
    sequence.window.pending_window.reset();
    sequence.window.pending_append.reset();
    if (!kvmem_window_tokens || !request.allow_memory_reuse || !sequence.window.stamp.valid ||
        (sequence.window.checkpoints.empty() && sequence.window.coverage.size()==0)) return;
    auto& window = sequence.window;
    if (window.session_key && !window.update_session_index) return;
    window.query_tokens=0;
    window.query_capture=false;
    for(auto& rows:window.query_rows) std::vector<float>().swap(rows);
    std::vector<float>().swap(window.query_sums);
    if (window.session_key) {
        for (const auto& publication : memory_publications) {
            if (publication.key == window.session_key && publication.order > window.publication_order) return;
        }
    }
    // Only the successful finish caller releases execution protection. Neither
    // query_replayed (set at rewind start) nor a late cancellation of a Finishable
    // request is proof that this is the successful publication path.
    if (completed && request.lifecycle == Lifecycle::Finishable &&
        !window.coverage.policy().extra_recompute_tokens &&
        memory_checkpoint(window, KvmemCheckpointKind::PromptTail)) {
        // T=0 preserves the previous single-Base retention footprint after the
        // temporary overlap needed to protect a currently executing replay.
        memory_checkpoint_slot(window, KvmemCheckpointKind::Base).reset();
    }
    if (completed && request.lifecycle == Lifecycle::Finishable && window.coverage.policy().extra_recompute_tokens) {
        try { prune_completed_memory_checkpoints(sequence); }
        catch (const std::exception& error) {
            std::fprintf(stderr, "KVMEM_CHECKPOINT_PRUNE skipped: %s\n", error.what());
        }
    }
    if(window.coverage.policy().enabled) {
      try {
        std::shared_ptr<KvmemPrefixCheckpoint> input;
        if (window.coverage.policy().extra_recompute_tokens ||
            memory_checkpoint(window, KvmemCheckpointKind::InputReuse)) {
            for (const auto kind : checkpoint_kinds(window)) {
                const auto& cp = memory_checkpoint_slot(window, kind);
                if (!cp || cp->prefix.size() >= window.output_begin ||
                    (input && cp->prefix.size() <= input->prefix.size()) ||
                    (cp->prefix.size() > window.query_begin && cp->prefix.size() < window.query_end) ||
                    !complete_memory_checkpoint(sequence, *cp)) continue;
                input = cp;
            }
            // Input-side reuse and append endpoints have distinct legal restore
            // limits. Prefer both under ordinary budgets; tiny budgets may evict.
            if (input) window.coverage.remember({kvmem::CheckpointPurpose::None, input}, kvmem::CoveragePurpose::Input);
        }
        // Reverse equal-frontier priority: the native Endpoint is published last.
        for(auto it=kKvmemCheckpointKinds.rbegin();it!=kKvmemCheckpointKinds.rend();++it) {
            const auto& cp=memory_checkpoint_slot(window,*it);
            if(cp) window.coverage.remember({kvmem::CheckpointPurpose::None,cp},
                *it==KvmemCheckpointKind::Endpoint ? kvmem::CoveragePurpose::Endpoint :
                *it==KvmemCheckpointKind::ReuseAnchor ? kvmem::CoveragePurpose::Anchor :
                cp==input ? kvmem::CoveragePurpose::Input : kvmem::CoveragePurpose::Periodic);
        }
        window.checkpoints.clear();
        window.coverage.prune();
      } catch(const std::exception& error) {
        // Retention is optional; never terminate a completed request if a
        // directory allocation fails. The active owner keeps its valid state.
        std::fprintf(stderr,"KVMEM_COVERAGE retention skipped: %s\n",error.what());
        return;
      }
    }
    std::uint32_t frontier = 0;
    for (const auto kind : checkpoint_kinds(window))
        if (const auto* checkpoint = memory_checkpoint(window, kind))
            frontier = std::max(frontier, static_cast<std::uint32_t>(checkpoint->prefix.size()));
    const auto pages = kv_pages_for_tokens(frontier);
    if (!pages || pages > window.archive.size() || pages > window.host_versions.size() ||
        !window.archive.front()) return;
    const auto page_bytes = window.archive.front().size();
    // All slots share one archive through the latest complete checkpoint.
    // Uncommitted/cancelled rows past that frontier are never retained.
    window.archive.resize(pages);
    window.host_versions.resize(pages);
    window.payload_bytes = pages * page_bytes;
    std::optional<std::size_t> destination;
    std::optional<std::size_t> oldest;
    std::size_t count = 0;
    for (std::size_t index = 0; index < memory_histories.size(); ++index) {
        auto& history = memory_histories[index];
        if (!history) { if (!destination) destination = index; continue; }
        ++count;
        if (window.session_key && history->session_key == window.session_key) {
            if (history->publication_order > window.publication_order) return;
            destination = index;
            break;
        }
        if (!oldest || history->publication_order < memory_histories[*oldest]->publication_order) oldest = index;
    }
    if (count >= kvmem_options.retained_sessions && (!destination || !memory_histories[*destination])) {
        destination = oldest;
        ++memory_history_evictions;
    }
    if (!destination) return;
    // Capture already completed every Host copy. The active state's later
    // writes and unpublished output never become this checkpoint's contents.
    for (auto& record : window.archive) record.seal();
    if(window.coverage.policy().enabled && kvmem_options.verify_transfers) {
        std::vector<const KvmemPrefixCheckpoint*> seen;
        for(const auto kind:checkpoint_kinds(window)) if(const auto* cp=memory_checkpoint(window,kind);
                cp && std::find(seen.begin(),seen.end(),cp)==seen.end()) {
            seen.push_back(cp);
            std::fprintf(stderr,"KVMEM_CHECKPOINT_DIRECTORY frontier=%zu cold=%d bytes=%llu identity=%p\n",
                cp->prefix.size(),int(!cp->state),(unsigned long long)(cp->state_bytes()+cp->metadata_bytes()),(const void*)cp);
        }
    }
    memory_histories[*destination].emplace(std::move(window));
    if (memory_io && memory_payload_storage.disk_bytes()) memory_idle_requested = true;

}

bool ProgramImpl::memory_maintenance_pending() const noexcept {
    return memory_io && (memory_idle_requested || memory_idle_transfer || memory_coverage_transfer);
}

bool ProgramImpl::progress_memory_storage() {
    if (!memory_maintenance_pending()) return false;
    bool changed = false;
    if(memory_coverage_transfer) {
        progress_checkpoint_coverage(continuation_states.front().window);
        if(memory_coverage_transfer) return false;
        changed = true;
    }
    const auto being_restored = [&](std::uint64_t generation) {
        const auto* transaction = std::get_if<MaterializationTransaction>(&context_transaction_);
        if (!transaction) return false;
        const auto& prefill = requests[transaction->destination.value].prefill;
        return prefill && prefill->memory_restore_frontier && prefill->memory_restore_generation == generation;
    };
    if (memory_idle_transfer) {
        auto& pending = *memory_idle_transfer;
        if (!pending.ticket.ready()) return false;
        try {
            pending.ticket.wait();
            auto& history = memory_histories.at(pending.history);
            if (history && history->stamp.session.id == pending.generation && !being_restored(pending.generation)) {
                auto& checkpoint = memory_checkpoint_slot(*history, pending.kind);
                if (checkpoint && checkpoint->state == pending.source && !checkpoint->cold_state.size()) {
                    // Fork metadata has its own lifetime, but an immutable native
                    // image gets one SSD extent even when several owners retain it.
                    for(auto& owner:memory_histories) if(owner && !being_restored(owner->stamp.session.id))
                        for(const auto kind:checkpoint_kinds(*owner)) {
                            auto& alias=memory_checkpoint_slot(*owner,kind);
                            if(!alias || alias->state!=pending.source || alias->cold_state) continue;
                            alias->cold_tag=pending.work->tag;
                            alias->cold_payload_bytes=pending.source->size();
                            alias->cold_state=pending.work->destination;
                            alias->state.reset();
                        }
                }
            }
        } catch (const std::exception& error) {
            std::fprintf(stderr, "KVMEM_IDLE state spill retained RAM: %s\n", error.what());
            memory_idle_requested = false; // retry on a subsequent retention boundary
        }
        memory_idle_transfer.reset(); // pinned owner is released on this engine thread
        changed = true;
    }
    if (!memory_idle_requested || has_context_transaction()) return changed;
    memory_idle_requested = false;
    const auto disk = memory_payload_storage.spill_file();
    std::uint64_t used = 0;
    for (const auto& active : requests) used += active.memory_payload_reservation;
    std::unordered_set<const void*> payloads,cold;
    for(const auto& lane:continuation_states) if(lane.window.statistics)
        for(const auto& record:lane.window.archive) if(record) payloads.insert(record.storage_identity());
    for (const auto& history : memory_histories) {
        if (!history) continue;
        for(const auto& record:history->archive)
            if(record && payloads.insert(record.storage_identity()).second) used+=record.size();
        std::vector<const KvmemPrefixCheckpoint*> seen;
        for (const auto kind : checkpoint_kinds(*history))
            if (const auto* checkpoint = memory_checkpoint(*history, kind);
                checkpoint && std::find(seen.begin(), seen.end(), checkpoint) == seen.end()) {
                if(checkpoint->cold_state && cold.insert(checkpoint->cold_state.storage_identity()).second)
                    used += checkpoint->cold_state.size();
                seen.push_back(checkpoint);
            }
    }
    // One background image at a time. Its original pinned owner is retained,
    // so there is no second full state-image allocation and no worker CUDA call.
    for (std::size_t index = 0; index < memory_histories.size(); ++index) {
        const auto& history = memory_histories[index];
        if (!history) continue;
        for (const auto kind : checkpoint_kinds(*history)) {
            const auto* checkpoint = memory_checkpoint(*history, kind);
            if (!checkpoint || !checkpoint->state || checkpoint->cold_state.size()) continue;
            const auto bytes = checkpoint->state->size();
            const auto disk_bytes = history->coverage.policy().enabled ? kvmem::checkpoint_image_size(bytes,1) : bytes;
            if (used > memory_retained_budget || disk_bytes > memory_retained_budget - used ||
                disk_bytes > disk->capacity_bytes() - disk->charged_bytes()) continue;
            try {
                auto pending = std::make_unique<KvmemIdleTransfer>();
                pending->history = index;
                pending->generation = history->stamp.session.id;
                pending->kind = kind;
                pending->source = checkpoint->state;
                pending->work = std::make_shared<KvmemIdleTransfer::Work>();
                pending->work->destination = disk->reserve(disk_bytes);
                pending->work->data = pending->source->data();
                pending->work->bytes = bytes;
                pending->work->tag = history->coverage.policy().enabled ? uint64_t(reinterpret_cast<uintptr_t>(checkpoint)) : 0;
                pending->ticket = memory_io->submit(bytes, [work = pending->work] {
                    if(work->tag) kvmem::write_checkpoint_image(work->destination,work->tag,{{work->data,work->bytes}});
                    else work->destination.write(0, work->data, work->bytes);
                });
                memory_idle_transfer = std::move(pending);
                memory_idle_requested = true;
                return true;
            } catch (const std::exception& error) {
                std::fprintf(stderr, "KVMEM_IDLE state spill retained RAM: %s\n", error.what());
                return changed;
            }
        }
    }
    return changed;
}

} // namespace ninfer::models::qwen3_5::detail
