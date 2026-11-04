#include "core/device.h"
#include "models/qwen3_5/program/storage/host_kv_store.h"
#include "models/qwen3_5/program/storage/kv_store.h"
#include "models/qwen3_5/program/storage/state_store.h"

#include "models/qwen3_5/state/state_image.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <new>
#include <string_view>
#include <vector>

namespace {

namespace q36   = ninfer::models::qwen3_5;
namespace store = ninfer::models::qwen3_5::detail;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

std::vector<std::int32_t> read_block_table(const ninfer::KVExecutionTablePool& tables,
                                           std::int32_t row, std::size_t count) {
    const ninfer::Tensor source = tables.matrix().slice(1, row, 1).view(
        {static_cast<std::int32_t>(tables.logical_page_capacity())});
    std::vector<std::int32_t> values(count);
    CUDA_CHECK(cudaMemcpy(values.data(), source.data, values.size() * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost));
    return values;
}

void test_state_store(ninfer::DeviceContext& device) {
    q36::StateImageSpec spec{
        .linear =
            {
                .layers         = 1,
                .conv_channels  = 8,
                .conv_width     = 3,
                .value_heads    = 2,
                .value_head_dim = 4,
                .key_head_dim   = 4,
                .slot_count     = 4,
                .conv_dtype     = ninfer::DType::BF16,
            },
        .hidden = 8,
        .dflash_local =
            q36::DFlashLocalStateSpec{.layers = 1, .capacity = 8, .kv_heads = 2, .head_dim = 4},
    };
    ninfer::LayoutBuilder builder;
    const q36::StateImageDeviceLayout layout = q36::plan_state_image_device_pool(builder, spec);
    ninfer::DeviceArena arena(builder.finish(256));
    q36::StateImageDevicePool physical({arena.base(), arena.capacity()}, layout);
    q36::HostStatePool host(layout.host, 2);
    store::StateImageStore images(
        physical, &host, static_cast<std::uint32_t>(physical.slot_count()) + host.capacity());

    const auto source = images.reserve_reset(device.stream);
    expect(source.has_value(), "state source allocation");
    const std::int32_t original_slot = images.physical_slot(*source);
    images.freeze(*source);
    images.move_checkpoint_to_active(*source);
    expect(images.physical_slot(*source) == original_slot &&
               images.role(*source) == store::StateImageRole::ActiveMutable,
           "private Move preserves the logical image and physical slot");
    images.freeze(*source);
    const auto destination = images.reserve_destination();
    expect(destination.has_value(), "state destination reservation");
    const store::StateImageSelectors selectors = images.begin_fork(*source, *destination);
    expect(selectors.source >= 0 && selectors.destination >= 0 &&
               selectors.source != selectors.destination,
           "fork resolves distinct physical selectors");
    expect(images.can_release_source_after_fork_abort(*source, *destination, 0) &&
               images.can_release_destination_after_fork_abort(*source, *destination, 0),
           "fork release preflight accounts for its source and destination pins");
    const auto concurrent_destination = images.reserve_destination();
    expect(concurrent_destination.has_value(), "concurrent fork destination reservation");
    (void)images.begin_fork(*source, *concurrent_destination);
    expect(!images.can_release_source_after_fork_abort(*source, *destination, 0),
           "fork release preflight preserves independent source pins");
    images.abort_fork(*source, *concurrent_destination);
    expect(images.release(*concurrent_destination), "concurrent fork destination release");
    expect(images.can_release_source_after_fork_abort(*source, *destination, 0),
           "fork release preflight accepts the final source pin");
    expect(!images.release(*source) && !images.release(*destination),
           "fork pins both logical images");
    images.commit_fork(*source, *destination);
    expect(images.role(*source) == store::StateImageRole::CheckpointImmutable &&
               images.role(*destination) == store::StateImageRole::ActiveMutable,
           "fork commit preserves source and activates destination");
    images.retain_checkpoint_reference(*source);
    const std::uint64_t rewrite_epoch = images.content_epoch(*source);
    images.freeze(*destination);
    const std::uint64_t recycled_epoch = images.recycle_checkpoint_destination(*source);
    const auto rotated_selectors       = images.begin_fork(*destination, *source);
    expect(rotated_selectors.source != rotated_selectors.destination && images.occupied() == 2,
           "rewrite rotation remains within two StateImages");
    images.abort_fork(*destination, *source);
    images.restore_recycled_checkpoint(*source, recycled_epoch);
    images.thaw(*destination);
    expect(images.content_epoch(*source) == rewrite_epoch &&
               images.checkpoint_references(*source) == 1,
           "aborted rewrite rotation restores the prior checkpoint identity");

    images.freeze(*destination);
    (void)images.recycle_checkpoint_destination(*source);
    (void)images.begin_fork(*destination, *source);
    images.commit_fork(*destination, *source);
    images.retain_checkpoint_reference(*destination);
    images.release_checkpoint_reference(*destination);
    expect(images.release(*destination) && images.release(*source) && images.occupied() == 0,
           "rotated state image ownership closes after release");
    expect(!images.valid(*source), "released state generation becomes stale");

    const auto reused = images.reserve_reset(device.stream);
    expect(reused.has_value() && *reused != *source, "state descriptor reuse advances generation");
    expect(images.release(*reused), "reused state image releases");

    const auto host_source = images.reserve_reset(device.stream);
    expect(host_source.has_value(), "Host state source allocation");
    device.synchronize();
    images.freeze(*host_source);
    auto d2h = images.begin_device_to_host(*host_source, device.transfer_stream);
    expect(d2h.has_value() &&
               images.residency(*host_source) == store::StateReplicaResidency::DeviceOnly,
           "incomplete State D2H does not publish a Host replica");
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    images.publish_transfer(std::move(*d2h), true);
    expect(images.residency(*host_source) == store::StateReplicaResidency::Both,
           "State D2H backup publishes stable Both residency");

    const auto moved_device = images.reserve_logical_destination();
    expect(moved_device.has_value(), "State replica split destination allocation");
    images.split_device_replica_identity(*host_source, *moved_device);
    expect(images.residency(*host_source) == store::StateReplicaResidency::HostOnly &&
               images.residency(*moved_device) == store::StateReplicaResidency::DeviceOnly,
           "State replica identity split keeps old Host content and moves Device ownership");

    const auto fork_one = images.reserve_logical_destination();
    const auto fork_two = images.reserve_logical_destination();
    expect(fork_one && fork_two, "concurrent Host State forks reserve logical destinations");
    auto h2d_one = images.begin_host_fork(*host_source, *fork_one, device.transfer_stream);
    auto h2d_two = images.begin_host_fork(*host_source, *fork_two, device.transfer_stream);
    expect(h2d_one && h2d_two && images.source_pins(*host_source) == 2,
           "immutable Host State source supports multiple fork pins");
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    images.publish_transfer(std::move(*h2d_one), true);
    images.publish_transfer(std::move(*h2d_two), true);
    expect(images.source_pins(*host_source) == 0 &&
               images.residency(*fork_one) == store::StateReplicaResidency::DeviceOnly &&
               images.residency(*fork_two) == store::StateReplicaResidency::DeviceOnly,
           "Host State forks publish independent Device destinations");
    expect(images.release(*host_source) && images.release(*moved_device) &&
               images.release(*fork_one) && images.release(*fork_two) && host.occupied() == 0,
           "State Host/Device replica ownership closes without leaked slots");
}

void test_kv_store(ninfer::DeviceContext& device) {
    ninfer::LayoutBuilder builder;
    ninfer::DeviceKVPagePoolSpec page_spec{
        .page_group_count = 8,
        .geometry =
            {
                .page_tokens        = static_cast<std::uint32_t>(ninfer::kPagedKVPageSize),
                .device_plane_order = ninfer::PagedKVPlaneOrder::PageMajor,
                .planes = {{.dtype = ninfer::DType::BF16, .leading_extent = 8, .head_extent = 2}},
            },
    };
    const ninfer::DeviceKVPagePoolLayout page_layout =
        ninfer::plan_device_kv_page_pool(builder, page_spec);
    const ninfer::KVExecutionTableLayout table_layout =
        ninfer::plan_kv_execution_tables(builder, {.logical_page_capacity = 4, .table_rows = 2});
    ninfer::DeviceArena arena(builder.finish(256));
    const ninfer::DeviceSpan backing{arena.base(), arena.capacity()};
    ninfer::DeviceKVPagePool physical_pages(backing, page_layout);
    ninfer::KVExecutionTablePool physical_tables(backing, table_layout, physical_pages);
    const ninfer::HostKVPageLayout host_layout =
        ninfer::plan_host_kv_page_layout(physical_pages.geometry());
    const std::array host_layouts{host_layout};
    ninfer::HostKVArena host_arena(host_layout.page_stride * 8, host_layouts);
    store::LogicalKVPageStore pages(physical_pages, physical_pages.capacity_pages() + 8U);
    store::HostKVExtentStore extents(host_arena, 8);
    store::KVAddressSpaceStore addresses(pages, physical_tables, 4, 4);

    const auto address = addresses.create_active(3, 0);
    expect(address.has_value(), "active KV address allocation");
    addresses.ensure_mapped_to_tokens(*address, 65, device.stream);
    device.synchronize();
    expect(addresses.mapped_pages(*address) == 2 && addresses.committed_frontier(*address) == 0,
           "physical KV append does not publish canonical coverage before commit");
    expect(read_block_table(physical_tables, 0, 2) == std::vector<std::int32_t>({0, 1}),
           "batched KV materialization publishes the exact execution mapping");
    addresses.commit_frontier(*address, 65);
    expect(addresses.mapped_pages(*address) == 2 && addresses.entitlement(*address) == 3 &&
               addresses.committed_frontier(*address) == 65 && addresses.bound_row(*address) == 0,
           "KV address tracks mapped pages, entitlement, frontier, and execution row");
    expect(pages.active_address_references(addresses.logical_page(*address, 0)) == 1 &&
               pages.active_address_references(addresses.logical_page(*address, 1)) == 1,
           "active KV membership is counted on each logical page");
    addresses.ensure_mapped_to_tokens(*address, 129, device.stream);
    device.synchronize();
    expect(read_block_table(physical_tables, 0, 3) == std::vector<std::int32_t>({0, 1, 2}),
           "incremental KV materialization preserves the existing mapping prefix");
    addresses.destructive_truncate(*address, 65);
    expect(addresses.mapped_pages(*address) == 2 && addresses.entitlement(*address) == 3 &&
               addresses.committed_frontier(*address) == 65,
           "same-frontier trim releases uncommitted materialized suffix pages");
    addresses.set_checkpoint_requirement(*address, 65);
    expect(pages.occupied() == 2 && addresses.mapped_pages(*address) == 2,
           "endpoint and rewrite requirements share one ordered page mapping");
    const std::uint64_t old_epoch = addresses.content_epoch(*address, 0);

    addresses.deactivate(*address);
    expect(addresses.bound_row(*address) == -1 && addresses.entitlement(*address) == 2,
           "catalogued KV address owns no execution row or growth reservation");

    const std::array logical_pages{addresses.logical_page(*address, 0),
                                   addresses.logical_page(*address, 1)};
    expect(pages.active_address_references(logical_pages[0]) == 0 &&
               pages.active_address_references(logical_pages[1]) == 0 &&
               !addresses.has_active_reference(logical_pages[0]),
           "KV deactivation clears logical-page active references");
    addresses.set_checkpoint_requirement(*address, 32);
    expect(pages.protected_columns(logical_pages[0]) == 32 &&
               pages.protected_columns(logical_pages[1]) == 0,
           "lowered checkpoint frontier releases stale suffix protection");
    addresses.set_checkpoint_requirement(*address, 65);
    auto host_backup = extents.prepare(pages, logical_pages);
    expect(host_backup.has_value(), "Host KV extent reservation");
    const std::vector<ninfer::DeviceKVPageHandle> sources = extents.device_sources(*host_backup);
    physical_pages.copy_to_host(sources, extents.writable_view(*host_backup),
                                device.transfer_stream);
    expect(!pages.host_resident(logical_pages[0]) && !pages.host_resident(logical_pages[1]),
           "incomplete KV D2H does not publish Host replicas");
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    auto host_extent = extents.publish(std::move(*host_backup));
    expect(pages.host_resident(logical_pages[0]) && pages.host_resident(logical_pages[1]),
           "KV extent publication attaches every logical Host replica");
    const std::array first_host_release{logical_pages[0]};
    expect(extents.release_page_replicas(pages, first_host_release),
           "first Host page release transaction");
    const auto retained_host_extent = pages.host_replica(logical_pages[1]).extent;
    expect(!extents.valid(host_extent) && !pages.host_resident(logical_pages[0]) &&
               extents.valid(retained_host_extent) &&
               pages.host_replica(logical_pages[1]).page_offset == 0,
           "partial Host release partitions an extent and republishes the retained run");
    const std::array second_host_release{logical_pages[1]};
    expect(extents.release_page_replicas(pages, second_host_release),
           "second Host page release transaction");
    expect(!pages.host_resident(logical_pages[1]) && host_arena.occupied_bytes() == 0,
           "partitioned Host replicas release while Device replicas survive");

    auto second_host_backup = extents.prepare(pages, logical_pages);
    expect(second_host_backup.has_value(), "second Host KV extent reservation");
    const std::vector<ninfer::DeviceKVPageHandle> second_sources =
        extents.device_sources(*second_host_backup);
    physical_pages.copy_to_host(second_sources, extents.writable_view(*second_host_backup),
                                device.transfer_stream);
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    const auto second_host_extent = extents.publish(std::move(*second_host_backup));
    expect(pages.drop_device_replica(logical_pages[0]) &&
               pages.drop_device_replica(logical_pages[1]) && !extents.release(second_host_extent),
           "Host-only KV pages remain valid and cannot lose their last replica");
    const std::array last_reference_release{store::HostKVPageReplicaRelease{
        .pages = &pages,
        .page  = logical_pages[1],
    }};
    const std::array seven_page_request{
        ninfer::HostKVAllocationRequest{.layout = &host_layout, .pages = 7}};
    const std::array eight_page_request{
        ninfer::HostKVAllocationRequest{.layout = &host_layout, .pages = 8}};
    expect(
        extents.can_allocate_after_page_releases({}, last_reference_release, seven_page_request) &&
            !extents.can_allocate_after_page_releases({}, last_reference_release,
                                                      eight_page_request),
        "last address-reference release contributes exact Host allocation credit");
    auto restore_reservation = physical_pages.reserve(2);
    expect(restore_reservation.has_value(), "KV Host restore Device reservation");
    const std::array restore_destinations{
        pages.reserve_device_replica(logical_pages[0], *restore_reservation),
        pages.reserve_device_replica(logical_pages[1], *restore_reservation)};
    physical_pages.copy_from_host(extents.view(second_host_extent), restore_destinations,
                                  device.transfer_stream);
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    pages.publish_device_replica(logical_pages[0]);
    pages.publish_device_replica(logical_pages[1]);
    expect(extents.release(second_host_extent) && host_arena.occupied_bytes() == 0,
           "KV Host restore republishes Device replicas before releasing the extent");
    auto activation = addresses.prepare_activation(*address, 3, 1);
    expect(addresses.bound_row(*address) == -1 && addresses.entitlement(*address) == 2 &&
               physical_pages.reserved_pages() == 1,
           "prepared KV activation preserves the catalogued mapping until publication");
    addresses.commit_activation(std::move(activation), device.stream);
    expect(pages.active_address_references(logical_pages[0]) == 1 &&
               pages.active_address_references(logical_pages[1]) == 1,
           "KV reactivation republishes logical-page active references");
    addresses.destructive_truncate(*address, 32);
    expect(addresses.mapped_pages(*address) == 1 && addresses.entitlement(*address) == 3 &&
               addresses.committed_frontier(*address) == 32 &&
               addresses.content_epoch(*address, 0) != old_epoch,
           "destructive rewrite truncates coverage and advances content epoch");
    addresses.deactivate(*address);
    const std::array final_logical_page{addresses.logical_page(*address, 0)};
    auto final_host_backup = extents.prepare(pages, final_logical_page);
    expect(final_host_backup.has_value(), "final Host KV backup reservation");
    const auto final_sources = extents.device_sources(*final_host_backup);
    physical_pages.copy_to_host(final_sources, extents.writable_view(*final_host_backup),
                                device.transfer_stream);
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    (void)extents.publish(std::move(*final_host_backup));
    expect(addresses.release(*address), "catalogued KV address releases");
    expect(extents.release_unreferenced() == host_layout.page_stride &&
               host_arena.occupied_bytes() == 0,
           "address teardown reclaims its zero-reference Host extent");
    expect(!addresses.valid(*address) && pages.occupied() == 0 &&
               physical_pages.allocated_pages() == 0 && physical_pages.reserved_pages() == 0,
           "KV release invalidates generations and closes physical ownership");

    // Verify crosses a page boundary, but the terminal commit consumes only its first column.
    const auto terminal = addresses.create_active(3, 0);
    expect(terminal.has_value(), "terminal boundary KV address allocation");
    addresses.ensure_mapped_to_tokens(*terminal, 63, device.stream);
    addresses.commit_frontier(*terminal, 63);
    addresses.ensure_mapped_to_tokens(*terminal, 71, device.stream);
    device.synchronize();
    const auto terminal_mapping   = read_block_table(physical_tables, 0, 2);
    const auto unchanged_terminal = [&] {
        return addresses.mapped_pages(*terminal) == 2 &&
               addresses.committed_frontier(*terminal) == 63 &&
               addresses.entitlement(*terminal) == 3 && physical_pages.allocated_pages() == 2 &&
               physical_pages.reserved_pages() == 1 && physical_pages.available_pages() == 5 &&
               read_block_table(physical_tables, 0, 2) == terminal_mapping;
    };
    addresses.ensure_mapped_to_tokens(*terminal, 71, device.stream);
    addresses.ensure_mapped_to_tokens(*terminal, 64, device.stream);
    device.synchronize();
    expect(unchanged_terminal(), "covered KV requests preserve speculative mappings and ownership");
    bool exceeded = false;
    try {
        addresses.ensure_mapped_to_tokens(*terminal, 193, device.stream);
    } catch (const std::invalid_argument&) { exceeded = true; }
    expect(exceeded && unchanged_terminal(),
           "KV coverage beyond entitlement fails without mutation");
    addresses.commit_frontier(*terminal, 64);
    addresses.destructive_truncate(*terminal, 64);
    expect(addresses.mapped_pages(*terminal) == 1 &&
               addresses.committed_frontier(*terminal) == 64 &&
               addresses.entitlement(*terminal) == 3 && physical_pages.allocated_pages() == 1 &&
               physical_pages.reserved_pages() == 2 && physical_pages.available_pages() == 5,
           "terminal trim returns the uncommitted page to the same active reservation");
    addresses.deactivate(*terminal);
    expect(addresses.release(*terminal) && physical_pages.allocated_pages() == 0 &&
               physical_pages.reserved_pages() == 0 && physical_pages.available_pages() == 8,
           "terminal settlement releases both mappings and unused growth");

    const auto snapshot_source      = addresses.create_active(3, 0);
    const auto snapshot_destination = addresses.create_inactive();
    expect(snapshot_source && snapshot_destination, "active KV snapshot endpoints allocate");
    addresses.ensure_mapped_to_tokens(*snapshot_source, 65, device.stream);
    addresses.commit_frontier(*snapshot_source, 65);
    const auto snapshot_full = addresses.logical_page(*snapshot_source, 0);
    const auto snapshot_tail = addresses.logical_page(*snapshot_source, 1);
    auto snapshot = addresses.prepare_active_snapshot(*snapshot_source, *snapshot_destination, 65);
    physical_pages.copy_page(addresses.active_snapshot_tail_source(snapshot),
                             addresses.active_snapshot_tail_destination(snapshot),
                             device.transfer_stream);
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    addresses.commit_active_snapshot(std::move(snapshot), device.stream);
    const auto snapshot_destination_tail = addresses.logical_page(*snapshot_destination, 1);
    expect(addresses.active(*snapshot_destination) && !addresses.active(*snapshot_source) &&
               pages.active_address_references(snapshot_full) == 1 &&
               pages.active_address_references(snapshot_tail) == 0 &&
               pages.active_address_references(snapshot_destination_tail) == 1,
           "active KV snapshot transfers active ownership without double-counting shared pages");
    addresses.deactivate(*snapshot_destination);
    expect(pages.active_address_references(snapshot_full) == 0 &&
               pages.active_address_references(snapshot_destination_tail) == 0 &&
               addresses.release(*snapshot_destination) && addresses.release(*snapshot_source) &&
               pages.occupied() == 0,
           "active KV snapshot references close with both address spaces");

    const auto alternating = addresses.create_active(4, 0);
    expect(alternating.has_value(), "alternating Host release address allocation");
    addresses.ensure_mapped_to_tokens(*alternating, 193, device.stream);
    addresses.commit_frontier(*alternating, 193);
    addresses.deactivate(*alternating);
    const std::array alternating_pages{
        addresses.logical_page(*alternating, 0), addresses.logical_page(*alternating, 1),
        addresses.logical_page(*alternating, 2), addresses.logical_page(*alternating, 3)};
    auto alternating_backup = extents.prepare(pages, alternating_pages);
    expect(alternating_backup.has_value(), "alternating Host extent reservation");
    const auto alternating_sources = extents.device_sources(*alternating_backup);
    physical_pages.copy_to_host(alternating_sources, extents.writable_view(*alternating_backup),
                                device.transfer_stream);
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    const auto alternating_extent = extents.publish(std::move(*alternating_backup));
    const std::array alternating_release{alternating_pages[0], alternating_pages[2]};
    expect(extents.release_page_replicas(pages, alternating_release),
           "alternating Host page release transaction");
    expect(!extents.valid(alternating_extent) && !pages.host_resident(alternating_pages[0]) &&
               pages.host_resident(alternating_pages[1]) &&
               !pages.host_resident(alternating_pages[2]) &&
               pages.host_resident(alternating_pages[3]) &&
               pages.host_replica(alternating_pages[1]).extent !=
                   pages.host_replica(alternating_pages[3]).extent &&
               host_arena.occupied_bytes() == 2U * host_layout.page_stride,
           "one batch partitions alternating Host release and retained runs exactly once");
    const std::array alternating_retained{alternating_pages[1], alternating_pages[3]};
    expect(extents.release_page_replicas(pages, alternating_retained),
           "retained Host run release transaction");
    expect(host_arena.occupied_bytes() == 0 && addresses.release(*alternating) &&
               pages.occupied() == 0,
           "alternating Host extent partitions close without leaked descriptors");

    const auto shared = addresses.create_active(3, 0);
    expect(shared.has_value(), "shared-prefix source address allocation");
    addresses.ensure_mapped_to_tokens(*shared, 65, device.stream);
    addresses.commit_frontier(*shared, 65);
    addresses.set_checkpoint_requirement(*shared, 65);
    addresses.deactivate(*shared);
    const auto shared_full = addresses.logical_page(*shared, 0);
    const auto shared_tail = addresses.logical_page(*shared, 1);

    const auto branch_one = addresses.create_inactive();
    expect(branch_one.has_value(), "first shared-prefix branch address allocation");
    auto first_fork = addresses.prepare_prefix_fork(*shared, *branch_one, 65, 3, 1);
    expect(first_fork.needs_tail_copy() && pages.address_references(shared_full) == 1,
           "prefix fork remains unpublished while its tail copy is pending");
    physical_pages.copy_page(addresses.prefix_fork_tail_source(first_fork),
                             addresses.prefix_fork_tail_destination(first_fork),
                             device.transfer_stream);
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    addresses.commit_prefix_fork(std::move(first_fork), device.stream);
    const auto branch_one_tail = addresses.logical_page(*branch_one, 1);
    expect(addresses.logical_page(*branch_one, 0) == shared_full &&
               branch_one_tail != shared_tail && pages.address_references(shared_full) == 2 &&
               pages.address_references(shared_tail) == 1,
           "shared branch references full pages and publishes a private partial tail");
    addresses.ensure_mapped_to_tokens(*branch_one, 66, device.stream);
    addresses.commit_frontier(*branch_one, 66);
    expect(pages.committed_columns(shared_tail) == 1 &&
               pages.committed_columns(branch_one_tail) == 2,
           "private branch writes do not extend the shared partial tail");
    addresses.deactivate(*branch_one);
    addresses.destructive_truncate_inactive(*branch_one, 65);
    expect(addresses.committed_frontier(*branch_one) == 65 &&
               pages.committed_columns(branch_one_tail) == 1 &&
               pages.address_references(shared_full) == 2,
           "private suffix truncation retains shared full-prefix pages");

    const auto branch_two = addresses.create_inactive();
    expect(branch_two.has_value(), "second shared-prefix branch address allocation");
    auto second_fork = addresses.prepare_prefix_fork(*shared, *branch_two, 65, 3, 1);
    physical_pages.copy_page(addresses.prefix_fork_tail_source(second_fork),
                             addresses.prefix_fork_tail_destination(second_fork),
                             device.transfer_stream);
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    addresses.commit_prefix_fork(std::move(second_fork), device.stream);
    const auto branch_two_tail = addresses.logical_page(*branch_two, 1);
    expect(branch_two_tail != shared_tail && branch_two_tail != branch_one_tail &&
               pages.address_references(shared_full) == 3,
           "independent shared branches own distinct partial tails and one shared full page");
    addresses.deactivate(*branch_two);
    expect(addresses.release(*branch_one) && pages.address_references(shared_full) == 2 &&
               addresses.release(*branch_two) && pages.address_references(shared_full) == 1 &&
               addresses.release(*shared) && pages.occupied() == 0 &&
               physical_pages.allocated_pages() == 0,
           "shared full-page occupancy survives until its final address reference releases");

    const auto mixed_source = addresses.create_active(4, 0);
    expect(mixed_source.has_value(), "mixed snapshot retained-prefix source allocation");
    addresses.ensure_mapped_to_tokens(*mixed_source, 65, device.stream);
    addresses.commit_frontier(*mixed_source, 65);
    addresses.set_checkpoint_requirement(*mixed_source, 65);
    addresses.deactivate(*mixed_source);
    const auto mixed_shared_full = addresses.logical_page(*mixed_source, 0);

    const auto mixed_active = addresses.create_inactive();
    expect(mixed_active.has_value(), "mixed snapshot active branch allocation");
    auto mixed_fork = addresses.prepare_prefix_fork(*mixed_source, *mixed_active, 65, 4, 0);
    physical_pages.copy_page(addresses.prefix_fork_tail_source(mixed_fork),
                             addresses.prefix_fork_tail_destination(mixed_fork),
                             device.transfer_stream);
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    addresses.commit_prefix_fork(std::move(mixed_fork), device.stream);
    addresses.ensure_mapped_to_tokens(*mixed_active, 130, device.stream);
    addresses.commit_frontier(*mixed_active, 130);
    const auto mixed_mutable_full = addresses.logical_page(*mixed_active, 1);
    const auto mixed_tail         = addresses.logical_page(*mixed_active, 2);
    const store::KVActiveSnapshotShape first_shape =
        addresses.active_snapshot_shape(*mixed_active, 130);
    expect(first_shape.full_pages == 2 && first_shape.immutable_full_pages == 1 &&
               first_shape.unique_full_pages == 1 && first_shape.tail_columns == 2 &&
               first_shape.copied_pages() == 1,
           "mixed snapshot analysis distinguishes aliased prefix, mutable suffix, and tail");

    const auto aborted_destination = addresses.create_inactive();
    expect(aborted_destination.has_value(), "mixed snapshot abort destination allocation");
    {
        auto aborted = addresses.prepare_active_snapshot(*mixed_active, *aborted_destination, 130);
        expect(pages.source_pins(mixed_shared_full) == 1 &&
                   pages.source_pins(mixed_mutable_full) == 1 && pages.source_pins(mixed_tail) == 1,
               "mixed snapshot preparation pins every source ownership class");
        addresses.abort_active_snapshot(aborted);
    }
    expect(addresses.active(*mixed_active) && pages.source_pins(mixed_shared_full) == 0 &&
               pages.source_pins(mixed_mutable_full) == 0 && pages.source_pins(mixed_tail) == 0 &&
               pages.writer_references(mixed_shared_full) == 0 &&
               pages.writer_references(mixed_mutable_full) == 1 &&
               pages.writer_references(mixed_tail) == 1 && addresses.release(*aborted_destination),
           "aborted mixed snapshot restores pins without changing page ownership");

    const auto first_snapshot_active = addresses.create_inactive();
    expect(first_snapshot_active.has_value(), "mixed snapshot destination allocation");
    auto first_mixed_snapshot =
        addresses.prepare_active_snapshot(*mixed_active, *first_snapshot_active, 130);
    physical_pages.copy_page(addresses.active_snapshot_tail_source(first_mixed_snapshot),
                             addresses.active_snapshot_tail_destination(first_mixed_snapshot),
                             device.transfer_stream);
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    addresses.commit_active_snapshot(std::move(first_mixed_snapshot), device.stream);
    const auto first_snapshot_tail = addresses.logical_page(*first_snapshot_active, 2);
    expect(
        !addresses.active(*mixed_active) && addresses.active(*first_snapshot_active) &&
            addresses.logical_page(*first_snapshot_active, 0) == mixed_shared_full &&
            addresses.logical_page(*first_snapshot_active, 1) == mixed_mutable_full &&
            first_snapshot_tail != mixed_tail && pages.address_references(mixed_shared_full) == 3 &&
            pages.address_references(mixed_mutable_full) == 2 &&
            pages.writer_references(mixed_shared_full) == 0 &&
            pages.writer_references(mixed_mutable_full) == 0 &&
            pages.writer_references(mixed_tail) == 0 &&
            pages.writer_references(first_snapshot_tail) == 1,
        "mixed snapshot reuses immutable full pages, freezes mutable full pages, and copies tail");

    addresses.ensure_mapped_to_tokens(*first_snapshot_active, 192, device.stream);
    addresses.commit_frontier(*first_snapshot_active, 192);
    const store::KVActiveSnapshotShape aligned_shape =
        addresses.active_snapshot_shape(*first_snapshot_active, 192);
    expect(aligned_shape.full_pages == 3 && aligned_shape.immutable_full_pages == 2 &&
               aligned_shape.unique_full_pages == 1 && aligned_shape.tail_columns == 0 &&
               aligned_shape.copied_pages() == 0,
           "repeated aligned snapshot analysis preserves the immutable-to-mutable boundary");
    const auto aligned_active = addresses.create_inactive();
    expect(aligned_active.has_value(), "aligned mixed snapshot destination allocation");
    auto aligned_snapshot =
        addresses.prepare_active_snapshot(*first_snapshot_active, *aligned_active, 192);
    expect(!aligned_snapshot.needs_tail_copy(), "aligned mixed snapshot requires no KV copy");
    addresses.commit_active_snapshot(std::move(aligned_snapshot), device.stream);
    addresses.ensure_mapped_to_tokens(*aligned_active, 193, device.stream);
    addresses.commit_frontier(*aligned_active, 193);
    expect(addresses.active(*aligned_active) && addresses.mapped_pages(*aligned_active) == 4 &&
               pages.writer_references(addresses.logical_page(*aligned_active, 3)) == 1,
           "aligned snapshot continuation materializes a new mutable suffix page");

    addresses.deactivate(*aligned_active);
    expect(addresses.release(*aligned_active) && addresses.release(*first_snapshot_active) &&
               addresses.release(*mixed_active) && addresses.release(*mixed_source) &&
               pages.occupied() == 0 && physical_pages.allocated_pages() == 0,
           "repeated mixed snapshot ownership closes without leaked logical or physical pages");

    const auto filler = addresses.create_active(4, 0);
    expect(filler.has_value(), "full-capacity staged-fork filler allocation");
    addresses.ensure_mapped_to_tokens(*filler, 193, device.stream);
    addresses.commit_frontier(*filler, 193);
    addresses.set_checkpoint_requirement(*filler, 193);
    addresses.deactivate(*filler);

    const auto retained = addresses.create_active(4, 0);
    expect(retained.has_value(), "full-capacity retained source allocation");
    addresses.ensure_mapped_to_tokens(*retained, 65, device.stream);
    addresses.commit_frontier(*retained, 65);
    addresses.set_checkpoint_requirement(*retained, 65);
    addresses.deactivate(*retained);
    const auto staged_destination = addresses.create_inactive();
    expect(staged_destination.has_value(), "full-capacity staged-fork destination allocation");
    expect(physical_pages.allocated_pages() == 6 && physical_pages.available_pages() == 2,
           "full-capacity staged-fork fixture leaves only the transient tail headroom");

    bool ordinary_fork_rejected = false;
    try {
        auto ordinary = addresses.prepare_prefix_fork(*retained, *staged_destination, 65, 4, 1);
        (void)ordinary;
    } catch (const std::bad_alloc&) { ordinary_fork_rejected = true; }
    expect(ordinary_fork_rejected,
           "ordinary retained prefix fork cannot reserve the one-page-over capacity shape");

    auto staged = addresses.prepare_prefix_fork(*retained, *staged_destination, 65, 4, 1, true);
    const auto retained_tail = addresses.prefix_fork_tail_logical_source(staged);
    physical_pages.copy_page(addresses.prefix_fork_tail_source(staged),
                             addresses.prefix_fork_tail_destination(staged),
                             device.transfer_stream);
    const std::array retained_tail_membership{retained_tail};
    auto retained_tail_backup = extents.prepare(pages, retained_tail_membership);
    expect(retained_tail_backup.has_value(), "retained tail Host backup reservation");
    const auto retained_tail_sources = extents.device_sources(*retained_tail_backup);
    physical_pages.copy_to_host(retained_tail_sources, extents.writable_view(*retained_tail_backup),
                                device.transfer_stream);
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    (void)extents.publish(std::move(*retained_tail_backup));
    addresses.settle_prefix_fork_tail_source(staged);
    expect(pages.drop_device_replica(retained_tail),
           "retained tail Device replica releases after both copies complete");
    addresses.complete_prefix_fork_after_tail_release(staged);
    addresses.commit_prefix_fork(std::move(staged), device.stream);
    expect(!pages.device_resident(retained_tail) && pages.host_resident(retained_tail) &&
               addresses.entitlement(*staged_destination) == 4 &&
               physical_pages.allocated_pages() == 6 && physical_pages.reserved_pages() == 2,
           "staged retained fork transfers the old tail capacity to the active entitlement");

    addresses.deactivate(*staged_destination);
    expect(addresses.release(*staged_destination) && addresses.release(*retained) &&
               addresses.release(*filler) &&
               extents.release_unreferenced() == host_layout.page_stride && pages.occupied() == 0 &&
               physical_pages.allocated_pages() == 0 && physical_pages.reserved_pages() == 0,
           "staged retained fork closes Device and Host ownership without leaks");
}

// Row -> physical publication under identity and hole windows (kvmem-port.md 2.12): a reselect
// may only move execution-table entries; Device pages never move and rows keep true-position
// order.
void test_kv_store_window(ninfer::DeviceContext& device) {
    ninfer::LayoutBuilder builder;
    ninfer::DeviceKVPagePoolSpec page_spec{
        .page_group_count = 8,
        .geometry =
            {
                .page_tokens        = static_cast<std::uint32_t>(ninfer::kPagedKVPageSize),
                .device_plane_order = ninfer::PagedKVPlaneOrder::PageMajor,
                .planes = {{.dtype = ninfer::DType::BF16, .leading_extent = 8, .head_extent = 2}},
            },
    };
    const ninfer::DeviceKVPagePoolLayout page_layout =
        ninfer::plan_device_kv_page_pool(builder, page_spec);
    const ninfer::KVExecutionTableLayout table_layout =
        ninfer::plan_kv_execution_tables(builder, {.logical_page_capacity = 6, .table_rows = 2});
    ninfer::DeviceArena arena(builder.finish(256));
    const ninfer::DeviceSpan backing{arena.base(), arena.capacity()};
    ninfer::DeviceKVPagePool physical_pages(backing, page_layout);
    ninfer::KVExecutionTablePool physical_tables(backing, table_layout, physical_pages);
    const ninfer::HostKVPageLayout host_layout =
        ninfer::plan_host_kv_page_layout(physical_pages.geometry());
    const std::array host_layouts{host_layout};
    ninfer::HostKVArena host_arena(host_layout.page_stride * 8, host_layouts);
    store::LogicalKVPageStore pages(physical_pages, physical_pages.capacity_pages() + 8U);
    store::HostKVExtentStore extents(host_arena, 8);
    store::KVAddressSpaceStore addresses(pages, physical_tables, 2, 6);

    const auto address = addresses.create_active(6, 0);
    expect(address.has_value(), "window KV address allocation");
    addresses.ensure_mapped_to_tokens(*address, 384, device.stream);
    addresses.commit_frontier(*address, 384);
    device.synchronize();
    const store::KVMemWindow default_view = addresses.window(*address);
    expect(default_view.is_identity(), "default KV window is the identity");
    expect(addresses.window_page_of(*address, 0) == 0 && addresses.window_page_of(*address, 3) == 3 &&
               addresses.window_page_of(*address, 5) == 5,
           "identity window pages equal logical pages");
    store::KVMemWindow identity_view;
    identity_view.block_tokens    = 128;
    identity_view.total_tokens    = 384;
    identity_view.window_tokens   = 384;
    identity_view.retained_pages  = 6;
    const std::vector<std::int32_t> identity = read_block_table(physical_tables, 0, 6);
    expect(identity == std::vector<std::int32_t>({0, 1, 2, 3, 4, 5}),
           "identity window publishes row equal to logical page");

    // A real two-block recency selection over three blocks yields the canonical sink + hole +
    // tail window, then the address republishes it without touching a Device page.
    store::KVMemBlockStore selection(store::KVMemConfig{.block_tokens = 128, .budget_tokens = 256});
    (void)selection.register_append(384);
    const store::KVMemPlan plan = selection.select_recency();
    const store::KVMemWindow hole = selection.window_for(plan, 383, 383);
    selection.apply(plan);
    expect(hole.has_hole() && hole.block_retained(0) && !hole.block_retained(1) &&
               hole.block_retained(2) && hole.window_tokens == 256 && hole.retained_pages == 4,
           "two-block budget over three blocks selects sink and tail around a hole");
    addresses.set_window(*address, hole, device.stream);
    device.synchronize();
    const std::vector<std::int32_t> holed = read_block_table(physical_tables, 0, 6);
    expect(holed[0] == identity[0] && holed[1] == identity[1] && holed[2] == identity[4] &&
               holed[3] == identity[5],
           "hole window keeps sink rows in place and slides the tail into the dense prefix");
    const store::KVMemWindow applied = addresses.window(*address);
    expect(applied.same_layout(hole) && applied.window_tokens == hole.window_tokens,
           "set window is recorded on the address");
    expect(addresses.window_page_of(*address, 1) == 1 && addresses.window_page_of(*address, 4) == 2 &&
               addresses.window_page_of(*address, 5) == 3,
           "window page renumbering matches rel() over the retained union");

    // Tier accounting: pages leaving the window release their active reference and count as the
    // host_only half of the claim; a snapshot hold round-trips that state; re-entry reclaims it.
    expect(addresses.demoted_pages(*address) == 2 && addresses.demoted_page(*address, 2) &&
               addresses.demoted_page(*address, 3) && !addresses.demoted_page(*address, 1) &&
               !addresses.demoted_page(*address, 4),
           "window exit pages count as demoted claims");
    const std::vector<std::uint32_t> held = addresses.hold_window_holes(*address);
    expect(held == std::vector<std::uint32_t>({2, 3}) && addresses.demoted_pages(*address) == 0 &&
               !addresses.demoted_page(*address, 2),
           "snapshot hold reclaims window holes for a uniformly held source");
    addresses.release_window_holes(*address, held);
    expect(addresses.demoted_pages(*address) == 2 && addresses.demoted_page(*address, 2) &&
               addresses.demoted_page(*address, 3),
           "snapshot hold release restores the demoted accounting");
    addresses.set_window(*address, identity_view, device.stream);
    device.synchronize();
    expect(addresses.demoted_pages(*address) == 0 && !addresses.demoted_page(*address, 2) &&
               read_block_table(physical_tables, 0, 6) == identity,
           "window re-entry reclaims ownership and republishes the identity");
    addresses.set_window(*address, hole, device.stream);
    device.synchronize();
    expect(addresses.demoted_pages(*address) == 2 && read_block_table(physical_tables, 0, 6) == holed,
           "window exit demotes again after a re-entry cycle");

    // Tier round trip at the store level: back the holes up to Host, drop their Device replicas
    // (what stage-out does), then restore through reserve_stage_in + publish (what stage-in does).
    const std::array hole_pages{addresses.logical_page(*address, 2),
                                addresses.logical_page(*address, 3)};
    auto hole_backup = extents.prepare(pages, hole_pages);
    expect(hole_backup.has_value(), "hole Host backup reservation");
    const std::vector<ninfer::DeviceKVPageHandle> hole_sources =
        extents.device_sources(*hole_backup);
    physical_pages.copy_to_host(hole_sources, extents.writable_view(*hole_backup),
                                device.transfer_stream);
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    const auto hole_extent = extents.publish(std::move(*hole_backup));
    expect(extents.valid(hole_extent), "hole extent publication");
    expect(pages.drop_device_replica(hole_pages[0]) && pages.drop_device_replica(hole_pages[1]),
           "stage-out drops Device replicas behind the Host backup");
    expect(!pages.device_resident(hole_pages[0]) && !pages.device_resident(hole_pages[1]) &&
               pages.host_resident(hole_pages[0]) && pages.host_resident(hole_pages[1]) &&
               addresses.demoted_pages(*address) == 2,
           "Host-only holes stay demoted after the drop");
    bool not_a_hole = false;
    try {
        (void)addresses.reserve_stage_in(*address, 0);
    } catch (const std::logic_error&) { not_a_hole = true; }
    expect(not_a_hole && !pages.device_resident(hole_pages[0]),
           "stage-in refuses a page that is not a window hole");
    (void)addresses.reserve_stage_in(*address, 2);
    expect(physical_pages.allocated_pages() == 5,
           "stage-in allocates the restored slot before publication");
    addresses.abort_stage_in(*address, 2);
    expect(physical_pages.allocated_pages() == 4 && physical_pages.reserved_pages() == 0 &&
               !pages.device_resident(hole_pages[0]),
           "aborted stage-in returns the pool claim");
    for (std::uint32_t index = 0; index < 2; ++index) {
        const auto destination = addresses.reserve_stage_in(*address, 2U + index);
        const store::HostKVPageReplica replica = pages.host_replica(hole_pages[index]);
        const std::array destinations{destination};
        physical_pages.copy_from_host(extents.view(replica.extent).subview(replica.page_offset, 1U),
                                      destinations, device.transfer_stream);
        pages.publish_device_replica(hole_pages[index]);
    }
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    expect(pages.device_resident(hole_pages[0]) && pages.device_resident(hole_pages[1]) &&
               addresses.demoted_pages(*address) == 2 && addresses.demoted_page(*address, 2),
           "stage-in restores Device residency while the demoted claim waits for reselect");
    const std::array first_hole_release{hole_pages[0]};
    const std::array second_hole_release{hole_pages[1]};
    expect(extents.release_page_replicas(pages, first_hole_release) &&
               extents.release_page_replicas(pages, second_hole_release),
           "restored hole Host replicas release");
    expect(host_arena.occupied_bytes() == 0, "hole Host extent frees after its replicas");

    bool misaligned = false;
    store::KVMemWindow misaligned_window = hole;
    misaligned_window.row_base_by_block[2] = 96;
    try {
        addresses.set_window(*address, misaligned_window, device.stream);
    } catch (const std::invalid_argument&) { misaligned = true; }
    expect(misaligned, "a block map that is not a dense page-aligned prefix is rejected");
    bool past_mapped = false;
    store::KVMemWindow past_mapped_window = hole;
    past_mapped_window.total_tokens = 448;
    try {
        addresses.set_window(*address, past_mapped_window, device.stream);
    } catch (const std::invalid_argument&) { past_mapped = true; }
    expect(past_mapped, "window past the mapped pages is rejected");
    bool overclaimed = false;
    store::KVMemWindow overclaimed_window = hole;
    overclaimed_window.window_tokens = 300;
    try {
        addresses.set_window(*address, overclaimed_window, device.stream);
    } catch (const std::invalid_argument&) { overclaimed = true; }
    expect(overclaimed, "window claiming more tokens than it retains is rejected");
    expect(read_block_table(physical_tables, 0, 6) == holed,
           "rejected windows leave the publication unchanged");

    // The window survives deactivation; reactivation republishes through the same window.
    addresses.deactivate(*address);
    expect(addresses.bound_row(*address) == -1, "windowed KV address drops its execution row");
    auto activation = addresses.prepare_activation(*address, 6, 0);
    addresses.commit_activation(std::move(activation), device.stream);
    device.synchronize();
    expect(addresses.bound_row(*address) == 0, "windowed KV reactivation rebinds an execution row");
    expect(read_block_table(physical_tables, 0, 6) == holed,
           "reactivation republishes through the retained window");

    addresses.deactivate(*address);
    expect(addresses.release(*address) && pages.occupied() == 0 &&
               physical_pages.allocated_pages() == 0 && physical_pages.reserved_pages() == 0,
           "windowed KV release closes physical ownership");

    // Destructive truncation over window holes on a fresh address: dropping fully demoted tail
    // pages releases their logical reference, a full boundary page that stays mapped is fine,
    // and a partial rewrite landing inside a hole is rejected.
    const auto truncated = addresses.create_active(6, 0);
    expect(truncated.has_value(), "truncate KV address allocation");
    addresses.ensure_mapped_to_tokens(*truncated, 384, device.stream);
    addresses.commit_frontier(*truncated, 384);
    device.synchronize();
    addresses.set_window(*truncated, hole, device.stream);
    device.synchronize();
    expect(addresses.demoted_pages(*truncated) == 2 && addresses.demoted_page(*truncated, 2) &&
               addresses.demoted_page(*truncated, 3),
           "second address demotes the same hole pages");
    addresses.destructive_truncate(*truncated, 256);
    expect(addresses.mapped_pages(*truncated) == 4 && addresses.demoted_pages(*truncated) == 2 &&
               addresses.committed_frontier(*truncated) == 256,
           "truncate past the hole drops retained tail pages and keeps demoted accounting");
    bool partial_hole = false;
    try {
        addresses.destructive_truncate(*truncated, 160);
    } catch (const std::logic_error&) { partial_hole = true; }
    expect(partial_hole && addresses.committed_frontier(*truncated) == 256 &&
               addresses.demoted_pages(*truncated) == 2,
           "partial rewrite into a hole page is rejected without state change");
    addresses.destructive_truncate(*truncated, 192);
    expect(addresses.mapped_pages(*truncated) == 3 && addresses.demoted_pages(*truncated) == 1 &&
               addresses.committed_frontier(*truncated) == 192,
           "full-boundary truncate through a hole releases the demoted page");
    addresses.destructive_truncate(*truncated, 128);
    expect(addresses.mapped_pages(*truncated) == 2 && addresses.demoted_pages(*truncated) == 0,
           "truncate drops the remaining demoted page at an aligned boundary");
    addresses.deactivate(*truncated);
    expect(addresses.release(*truncated) && pages.occupied() == 0 &&
               physical_pages.allocated_pages() == 0 && physical_pages.reserved_pages() == 0,
           "truncated windowed KV release closes physical ownership");

    // Truncation resets the window to the identity sentinel while the published table still holds
    // the hole layout. The next identity reselect must therefore republish instead of trusting
    // same_layout(), or the address would run on a stale renumbering.
    const auto reset = addresses.create_active(6, 0);
    expect(reset.has_value(), "reset KV address allocation");
    addresses.ensure_mapped_to_tokens(*reset, 384, device.stream);
    addresses.commit_frontier(*reset, 384);
    addresses.set_window(*reset, hole, device.stream);
    device.synchronize();
    expect(read_block_table(physical_tables, 0, 6) == holed,
           "reset address publishes the hole window");
    addresses.destructive_truncate(*reset, 256);
    store::KVMemWindow reset_identity;
    reset_identity.block_tokens   = 128;
    reset_identity.total_tokens   = 256;
    reset_identity.window_tokens  = 256;
    reset_identity.retained_pages = 4;
    addresses.set_window(*reset, reset_identity, device.stream);
    device.synchronize();
    expect(addresses.demoted_pages(*reset) == 0 &&
               read_block_table(physical_tables, 0, 4) ==
                   std::vector<std::int32_t>(identity.begin(), identity.begin() + 4),
           "identity reselect after a truncation republishes the dense table");
    addresses.deactivate(*reset);
    expect(addresses.release(*reset) && physical_pages.allocated_pages() == 0,
           "reset KV release closes physical ownership");
}

// A context larger than the Device pool (--kvmem-budget / ring shape): the logical entitlement is
// what bounds coverage, the pool claim is what the pool is charged, and pages the window drops are
// what makes the next round's growth possible. This is the store half of a windowed request.
void test_kv_store_staged_context(ninfer::DeviceContext& device) {
    ninfer::LayoutBuilder builder;
    ninfer::DeviceKVPagePoolSpec page_spec{
        .page_group_count = 6,
        .geometry =
            {
                .page_tokens        = static_cast<std::uint32_t>(ninfer::kPagedKVPageSize),
                .device_plane_order = ninfer::PagedKVPlaneOrder::PageMajor,
                .planes = {{.dtype = ninfer::DType::BF16, .leading_extent = 8, .head_extent = 2}},
            },
    };
    const ninfer::DeviceKVPagePoolLayout page_layout =
        ninfer::plan_device_kv_page_pool(builder, page_spec);
    const ninfer::KVExecutionTableLayout table_layout =
        ninfer::plan_kv_execution_tables(builder, {.logical_page_capacity = 12, .table_rows = 1});
    ninfer::DeviceArena arena(builder.finish(256));
    const ninfer::DeviceSpan backing{arena.base(), arena.capacity()};
    ninfer::DeviceKVPagePool physical_pages(backing, page_layout);
    ninfer::KVExecutionTablePool physical_tables(backing, table_layout, physical_pages);
    const ninfer::HostKVPageLayout host_layout =
        ninfer::plan_host_kv_page_layout(physical_pages.geometry());
    const std::array host_layouts{host_layout};
    ninfer::HostKVArena host_arena(host_layout.page_stride * 8, host_layouts);
    store::LogicalKVPageStore pages(physical_pages, physical_pages.capacity_pages() + 8U);
    store::HostKVExtentStore extents(host_arena, 8);
    store::KVAddressSpaceStore addresses(pages, physical_tables, 1, 12);

    // Twelve pages of entitlement on a six-page pool: activation cannot claim the logical gap, so
    // it claims what the pool has and coverage stays bounded by the logical entitlement.
    const auto address = addresses.create_active(12, 0);
    expect(address.has_value(), "staged KV address allocation");
    expect(addresses.entitlement(*address) == 12 &&
               addresses.claimed_tail_pages(*address) == physical_pages.capacity_pages() &&
               physical_pages.reserved_pages() == physical_pages.capacity_pages(),
           "activation of an oversized context claims the pool, not the logical entitlement");

    // Growth is progressive: each round maps only what it runs, drawing on the claim.
    addresses.ensure_mapped_to_tokens(*address, 128, device.stream);
    addresses.commit_frontier(*address, 128);
    device.synchronize();
    expect(addresses.mapped_pages(*address) == 2 &&
               addresses.claimed_tail_pages(*address) == physical_pages.capacity_pages() - 2 &&
               physical_pages.allocated_pages() == 2,
           "mapped pages move from the claim to the allocation without changing the total");
    addresses.ensure_mapped_to_tokens(*address, 320, device.stream);
    addresses.commit_frontier(*address, 320);
    device.synchronize();
    expect(addresses.mapped_pages(*address) == 5 && physical_pages.allocated_pages() == 5 &&
               addresses.claimed_tail_pages(*address) == 1,
           "coverage grows while the claim covers the rest of the pool");
    addresses.ensure_mapped_to_tokens(*address, 384, device.stream);
    addresses.commit_frontier(*address, 384);
    device.synchronize();
    expect(addresses.mapped_pages(*address) == 6 &&
               addresses.claimed_tail_pages(*address) == 0,
           "a full working set leaves the pool claim empty");

    // A two-block budget over three blocks keeps the sink block and the newest one, so the middle
    // block leaves the window. Its Host backup plus drop is what stage-out does, and it is the only
    // way a context larger than the pool keeps mapping forward.
    store::KVMemBlockStore selection(store::KVMemConfig{.block_tokens = 128, .budget_tokens = 256});
    (void)selection.register_append(384);
    store::KVMemPlan plan = selection.select_recency();
    store::KVMemWindow window = selection.window_for(plan, 319, 319);
    selection.apply(plan);
    expect(window.block_retained(0) && !window.block_retained(1) && window.block_retained(2) &&
               window.window_tokens == 256 && window.retained_pages == 4,
           "two-block budget over three blocks keeps sink and newest block");
    addresses.set_window(*address, window, device.stream);
    device.synchronize();
    expect(addresses.demoted_pages(*address) == 2 && addresses.demoted_page(*address, 2) &&
               addresses.demoted_page(*address, 3) && !addresses.demoted_page(*address, 4),
           "window exit pages are demoted but stay mapped");
    const std::array first_exit{addresses.logical_page(*address, 2),
                                addresses.logical_page(*address, 3)};
    auto backup = extents.prepare(pages, first_exit);
    expect(backup.has_value(), "staged exit Host backup");
    const std::vector<ninfer::DeviceKVPageHandle> sources = extents.device_sources(*backup);
    physical_pages.copy_to_host(sources, extents.writable_view(*backup), device.transfer_stream);
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    const auto extent = extents.publish(std::move(*backup));
    expect(extent.valid(), "staged exit Host extent publication");
    for (const auto page : first_exit) {
        expect(pages.drop_device_replica(page), "stage-out drops the exit Device replica");
    }
    expect(physical_pages.allocated_pages() == 4 && physical_pages.available_pages() == 2 &&
               !pages.device_resident(first_exit[0]),
           "dropped window exits return their pages to the pool");

    // The next round's growth comes out of the freed pages: the claim had to be capped at
    // activation, because the pages this request will eventually map exceed the pool.
    addresses.ensure_mapped_to_tokens(*address, 448, device.stream);
    addresses.commit_frontier(*address, 448);
    device.synchronize();
    expect(addresses.mapped_pages(*address) == 7 && physical_pages.allocated_pages() == 5 &&
               addresses.claimed_tail_pages(*address) == 1,
           "a capped claim tops up from staged-out pages as the window advances");

    // Second slide: the window moves over a partially appended context, so the retained tail is the
    // newest partial block and a different page pair leaves.
    (void)selection.register_append(64);
    plan                = selection.select_recency();
    const store::KVMemWindow second_window = selection.window_for(plan, 447, 447);
    selection.apply(plan);
    expect(second_window.block_retained(0) && !second_window.block_retained(1) &&
               !second_window.block_retained(2) && second_window.block_retained(3) &&
               second_window.window_tokens == 192 && second_window.retained_pages == 3,
           "the window follows the frontier into a partial trailing block");
    addresses.set_window(*address, second_window, device.stream);
    device.synchronize();
    const std::array second_exit{addresses.logical_page(*address, 4),
                                 addresses.logical_page(*address, 5)};
    expect(addresses.demoted_pages(*address) == 4 && pages.device_resident(second_exit[0]),
           "the next block pair leaves the window still Device-resident");
    auto second_backup = extents.prepare(pages, second_exit);
    expect(second_backup.has_value(), "second staged exit Host backup");
    const std::vector<ninfer::DeviceKVPageHandle> second_sources =
        extents.device_sources(*second_backup);
    physical_pages.copy_to_host(second_sources, extents.writable_view(*second_backup),
                                device.transfer_stream);
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    const auto second_extent = extents.publish(std::move(*second_backup));
    for (const auto page : second_exit) {
        expect(pages.drop_device_replica(page), "second stage-out drops its Device replica");
    }
    addresses.ensure_mapped_to_tokens(*address, 512, device.stream);
    addresses.commit_frontier(*address, 512);
    device.synchronize();
    expect(addresses.mapped_pages(*address) == 8 &&
               addresses.mapped_pages(*address) > physical_pages.capacity_pages() &&
               addresses.claimed_tail_pages(*address) == 0 &&
               physical_pages.allocated_pages() == 4,
           "coverage exceeds the pool while resident pages stay inside it");

    // A pool with nothing left to hand out is an honest failure, not a silent overshoot.
    bool exhausted = false;
    try {
        addresses.ensure_mapped_to_tokens(*address, 768, device.stream);
    } catch (const std::bad_alloc&) { exhausted = true; }
    expect(exhausted && addresses.mapped_pages(*address) == 8 &&
               addresses.entitlement(*address) == 12,
           "growing past what the pool can back fails without changing coverage");

    addresses.deactivate(*address);
    const bool released = addresses.release(*address);
    expect(released && physical_pages.allocated_pages() == 0 && physical_pages.reserved_pages() == 0,
           "staged KV release closes physical ownership");
    // Zero-reference logical descriptors of pages with Host replicas stay alive by design until
    // the extent store detaches the replica as one atomic allocation (release_reference).
    const std::size_t freed = extents.release_unreferenced();
    expect(freed == 4 * host_layout.page_stride && host_arena.occupied_bytes() == 0 &&
               pages.occupied() == 0,
           "staged-out Host backups free once nothing references them");
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || count == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    CUDA_CHECK(count_err);

    try {
        ninfer::DeviceContext device(0);
        test_state_store(device);
        test_kv_store(device);
        test_kv_store_window(device);
        test_kv_store_staged_context(device);
        device.synchronize();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
        return 1;
    }
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
