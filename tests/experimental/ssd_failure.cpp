#include "ninfer/engine.h"
#include "kvmem_test_options.h"
#include "kvmem/host_kv_storage.hpp"
#include "kvmem/spill_io.hpp"
#include "kvmem/memory_contract.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>

#ifdef _WIN32
#include <windows.h>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

// Replace only this test executable's ReadFile/WriteFile import. No production hook,
// environment variable, filesystem modification or other process is involved.
// One KV write receives the same error as a full Windows volume would return.
template<bool Reading = false> class FileFailure {
    using Pointer = std::conditional_t<Reading, LPVOID, LPCVOID>;
    using Write = BOOL(WINAPI*)(HANDLE, Pointer, DWORD, LPDWORD, LPOVERLAPPED);
    inline static Write original = nullptr;
    inline static std::atomic<bool> armed{false};
    inline static std::atomic<unsigned> failures{0};
    inline static std::mutex gate_mutex;
    inline static std::condition_variable gate_changed;
    inline static bool hold = false, entered = false, released = false;
    ULONG_PTR* slot = nullptr;
    static BOOL WINAPI write(HANDLE file, Pointer data, DWORD size, LPDWORD done, LPOVERLAPPED op) {
        if (armed.load()) {
            wchar_t path[2048]{};
            const auto count = GetFinalPathNameByHandleW(file, path, 2048, FILE_NAME_NORMALIZED);
            if (count && count < 2048 && std::wcsstr(path, L"kvmem-spill-") && armed.exchange(false)) {
                std::unique_lock<std::mutex> lock(gate_mutex);
                if (hold) {
                    entered = true;
                    gate_changed.notify_all();
                    gate_changed.wait(lock, [] { return released; });
                    lock.unlock();
                    return original(file, data, size, done, op);
                }
                ++failures;
                SetLastError(Reading ? ERROR_READ_FAULT : ERROR_DISK_FULL);
                return FALSE;
            }
        }
        return original(file, data, size, done, op);
    }
    void replace(ULONG_PTR value) {
        DWORD previous = 0;
        require(VirtualProtect(slot, sizeof(*slot), PAGE_READWRITE, &previous) != 0, "cannot change test import");
        *slot = value;
        DWORD ignored = 0;
        require(VirtualProtect(slot, sizeof(*slot), previous, &ignored) != 0, "cannot restore test import protection");
    }
public:
    FileFailure() {
        auto* base = reinterpret_cast<unsigned char*>(GetModuleHandleW(nullptr));
        auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(base);
        auto* nt = reinterpret_cast<IMAGE_NT_HEADERS*>(base + dos->e_lfanew);
        auto* imports = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(base +
            nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT].VirtualAddress);
        for (; imports->Name && !slot; ++imports) {
            if (!imports->OriginalFirstThunk) continue;
            auto* names = reinterpret_cast<IMAGE_THUNK_DATA*>(base + imports->OriginalFirstThunk);
            auto* addresses = reinterpret_cast<IMAGE_THUNK_DATA*>(base + imports->FirstThunk);
            for (; names->u1.AddressOfData; ++names, ++addresses) {
                if (IMAGE_SNAP_BY_ORDINAL(names->u1.Ordinal)) continue;
                auto* name = reinterpret_cast<IMAGE_IMPORT_BY_NAME*>(base + names->u1.AddressOfData);
                if (std::strcmp(reinterpret_cast<const char*>(name->Name), Reading ? "ReadFile" : "WriteFile") == 0) {
                    slot = &addresses->u1.Function;
                    original = reinterpret_cast<Write>(*slot);
                    break;
                }
            }
        }
        require(slot, "test executable does not import required file operation");
        replace(reinterpret_cast<ULONG_PTR>(&write));
    }
    ~FileFailure() { armed = false; release(); if (slot) replace(reinterpret_cast<ULONG_PTR>(original)); }
    void arm(bool block = false) {
        std::lock_guard<std::mutex> lock(gate_mutex);
        hold = block; entered = released = false; armed = true;
    }
    bool wait_entered() {
        std::unique_lock<std::mutex> lock(gate_mutex);
        return gate_changed.wait_for(lock, std::chrono::seconds(30), [] { return entered; });
    }
    void release() {
        std::lock_guard<std::mutex> lock(gate_mutex);
        released = true; gate_changed.notify_all();
    }
    unsigned count() const { return failures.load(); }
};
using WriteFailure = FileFailure<false>;

void cold_rewrite_peer_progress() {
    kvmem::HostKvStorage storage(32, std::make_shared<kvmem::SpillFile>(
        std::filesystem::current_path() / "promotion-spill", 128));
    const unsigned char expected[8] = {1,2,3,4,5,6,7,8};
    auto cold = storage.allocate(8, expected);
    cold.seal(); storage.trim(0);
    FileFailure<true> gate;
    gate.arm(true);
    auto rewrite = std::async(std::launch::async, [&] { cold.writable_data()[0] = 9; });
    const bool entered = gate.wait_entered();
    auto peer = std::async(std::launch::async, [&] { return storage.allocate(8); });
    const bool progressed = peer.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
    gate.release(); // Release before assertions/future destruction even on failure.
    rewrite.get(); auto resident = peer.get();
    require(entered && progressed, "cold rewrite blocked an independent RAM reservation");
    cold.seal(); storage.trim(0);
    gate.arm();
    bool failed = false;
    try { (void)cold.writable_data(); }
    catch (const kvmem::MemoryError& error) { failed = error.code() == kvmem::MemoryErrorCode::TransferFailed; }
    require(failed && cold.resident_bytes() == 0 && storage.charged_bytes() == 8,
            "failed promotion discarded disk or leaked reserved RAM");
    unsigned char restored[8]{};
    cold.read(0, restored, sizeof(restored));
    require(restored[0] == 9 && std::memcmp(restored+1, expected+1, 7) == 0,
            "failed promotion corrupted the only valid copy");
    std::cout << "COLD_REWRITE peer progress and failed-read rollback PASS" << std::endl;
}
struct Sink final : ninfer::OutputSink {
    std::promise<void> first;
    bool notified = false;
    std::atomic<unsigned> generated{0};
    void start(ninfer::GenerationStart) override {}
    void progress(ninfer::PromptProgress) override {}
    void publish(ninfer::OutputDelta) override {}
    void timing(ninfer::GenerationTimingObservation value) override {
        generated = value.generated_tokens;
        if (value.generated_tokens && !notified) { notified = true; first.set_value(); }
    }
};
ninfer::PromptInput prompt(bool cold) {
    ninfer::PromptInput input;
    input.options.enable_thinking = false;
    input.context_cache.session_key = cold ? "cold-failure" : "hot-peer";
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    std::string text;
    if (cold) for (int i = 0; i < 160; ++i)
        text += "The observatory archives pulsars, cosmic rays and distant galaxies. ";
    text += "List 100 numbered facts about astronomy.";
    message.parts.push_back({.kind = ninfer::MessagePartKind::Text, .text = std::move(text)});
    input.messages.push_back(std::move(message));
    return input;
}
ninfer::RequestOptions request(unsigned output) {
    ninfer::RequestOptions value;
    value.execution.requested_output_tokens = output;
    value.execution.sampling.temperature = 0;
    value.stop.include_model_defaults = false;
    return value;
}

void checkpoint_fork(const char* artifact) {
    _putenv_s("KVMEM_CHECKPOINT_POLICY", "spread");
    _putenv_s("KVMEM_CHECKPOINT_HOT", "1");
    ninfer::EngineOptions options;
    ninfer::test::kvmem_test_devices(options);
    options.artifact_path=artifact;
    options.device_profile="off";
    options.max_context=8192;
    options.prefill_chunk=128;
    options.kv_cache=ninfer::KvCacheStorage::Int8Group64;
    options.kv_capacity=ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.kvmem={.selected_tokens=3968,.reserve_tokens=128,.host_bytes=256ULL<<20,
        .retained_sessions=4,.verify_transfers=true};
    // This fixture retains several owners and full native StateImages. On 27B,
    // one image is about 147 MiB: the old 2 GiB quota filled before the cancel
    // case, leaving the selected checkpoint hot instead of exercising ReadFile.
    // This is an upper bound; SpillFile allocates/writes extents on demand.
    options.kvmem.disk_bytes=8ULL<<30;
    options.kvmem.disk_path=std::filesystem::current_path()/"fork-fault-spill";
    const auto available_disk=std::filesystem::space(std::filesystem::current_path()).available;
    require(available_disk>options.kvmem.disk_bytes,"fork fixture lacks SSD quota headroom");
    std::cout<<"FORK_STORAGE quota="<<options.kvmem.disk_bytes
             <<" available="<<available_disk<<std::endl;
    options.context_cache.enabled=false;
    ninfer::PromptInput document;
    document.options.enable_thinking=false;
    document.context_cache.session_key="parent";
    const auto add=[](ninfer::PromptInput& input,ninfer::ChatRole role,std::string text) {
        ninfer::ChatMessage message; message.role=role;
        message.parts.push_back({.kind=ninfer::MessagePartKind::Text,.text=std::move(text)});
        input.messages.push_back(std::move(message));
    };
    add(document,ninfer::ChatRole::System,"Answer briefly using the supplied reference.");
    std::string text="Reference: The violet observatory studies pulsars. Its access code is MARIGOLD. ";
    for(int i=0;i<80;++i) text+="The coastal office files invoices, transport schedules and warehouse inventories. ";
    text+="Acknowledge this reference with only Recorded. The question comes later.";
    add(document,ninfer::ChatRole::User,text);
    {
        ninfer::Engine engine(options);
        const auto image_bytes=engine.memory_summary().host_state_image_bytes;
        require(image_bytes>0,"fork fixture requires native StateImages");
        std::uint64_t observed_disk_payload=0;
        const auto wait_cold=[&](const char* stage) {
            // Idle maintenance demotes all retained images, including execution
            // slots. The checkpoint pool caches one released image; metadata is
            // still counted. Wait for that bounded idle state rather than assume
            // a one-second sleep makes the next owner's restore cold. ReadFile's
            // entered gate below remains the decisive proof of a cold restore.
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
            auto quiet_since=std::chrono::steady_clock::now();
            auto memory=engine.memory_summary().kvmem;
            while(std::chrono::steady_clock::now()<deadline) {
                auto next=engine.memory_summary().kvmem;
                observed_disk_payload=std::max(observed_disk_payload,next.disk_payload_bytes);
                if(next.checkpoint_bytes!=memory.checkpoint_bytes ||
                   next.disk_written_bytes!=memory.disk_written_bytes)
                    quiet_since=std::chrono::steady_clock::now();
                memory=next;
                require(memory.disk_payload_bytes<=options.kvmem.disk_bytes &&
                        memory.host_payload_bytes<=options.kvmem.host_bytes,
                        "fork preparation exceeded storage quota");
                if(!memory.active_sessions && memory.disk_payload_bytes>=image_bytes &&
                   memory.checkpoint_bytes<2*image_bytes &&
                   std::chrono::steady_clock::now()-quiet_since>=std::chrono::milliseconds(200)) {
                    std::cout<<"FORK_COLD_READY stage="<<stage<<" image_bytes="<<image_bytes
                             <<" checkpoint_ram="<<memory.checkpoint_bytes
                             <<" disk_payload="<<memory.disk_payload_bytes
                             <<" written="<<memory.disk_written_bytes<<std::endl;
                    return memory;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            std::cerr<<"FORK_COLD_NOT_READY stage="<<stage<<" image_bytes="<<image_bytes
                     <<" checkpoint_ram="<<memory.checkpoint_bytes
                     <<" disk_payload="<<memory.disk_payload_bytes
                     <<" quota="<<options.kvmem.disk_bytes<<std::endl;
            throw std::runtime_error("fork fixture did not drain retained StateImages to SSD");
        };
        auto ack_options=request(8); ack_options.stop.include_model_defaults=true;
        const auto ack=engine.submit(engine.prepare(document),ack_options).wait();
        auto question=document;
        add(question,ninfer::ChatRole::Assistant,ack.content);
        add(question,ninfer::ChatRole::User,"What is the violet observatory access code? Reply with just the code.");
        auto cold=question; cold.context_cache.session_key="reference";
        auto response_options=request(12); response_options.stop.include_model_defaults=true;
        auto fresh=response_options; fresh.execution.allow_prefix_reuse=false;
        const auto reference=engine.submit(engine.prepare(cold),fresh).wait();
        require(!reference.reused_prompt_tokens,"reference unexpectedly reused a prefix");
        const auto answer_begin=reference.content.find_first_not_of(" \t\r\n");
        const auto answer_end=reference.content.find_last_not_of(" \t\r\n");
        require(answer_begin!=std::string::npos &&
            reference.content.substr(answer_begin,answer_end-answer_begin+1)=="MARIGOLD",
            "fresh prefill missed the independent reference answer");
        const auto run=[&](const char* key,bool expected_hit=true) {
            auto input=question; input.context_cache.session_key=key;
            auto result=engine.submit(engine.prepare(input),response_options).wait();
            require(!expected_hit || result.reused_prompt_tokens>0,"fork or parent missed its checkpoint");
            std::cout<<"FORK_OUTPUT "<<key<<" reference="<<reference.content<<" restored="<<result.content<<"\nreference_ids:";
            for(auto id:reference.generated_token_ids) std::cout<<' '<<id;
            std::cout<<"\nrestored_ids:";for(auto id:result.generated_token_ids) std::cout<<' '<<id;
            std::cout<<std::endl;
            require(result.generated_token_ids==reference.generated_token_ids,"fork/fallback output differs from fresh prefill");
            require(engine.is_available(),"fork disabled Engine");
            const auto memory=engine.memory_summary().kvmem;
            observed_disk_payload=std::max(observed_disk_payload,memory.disk_payload_bytes);
            require(memory.host_payload_bytes<=options.kvmem.host_bytes &&
                memory.disk_payload_bytes<=options.kvmem.disk_bytes && !memory.resident_pages,
                "fork leaked quota or native pages");
            std::cout<<"FORK_CASE "<<key<<" reused="<<result.reused_prompt_tokens
                     <<" native_hits="<<memory.history_hits<<" reads="<<memory.disk_read_bytes
                     <<" errors="<<memory.disk_errors<<" disk_payload="<<memory.disk_payload_bytes
                     <<" exact=1"<<std::endl;
            return result;
        };
        wait_cold("initial-fork");
        run("child"); run("parent");
        wait_cold("read-failure");
        FileFailure<true> failure;
        const auto before=engine.memory_summary().kvmem;
        failure.arm();
        run("failed-child",false);
        const auto failed=engine.memory_summary().kvmem;
        require(failure.count()==1 && failed.disk_errors==before.disk_errors+1,
                "fork did not exercise injected cold read failure");
        require(failed.history_hits==before.history_hits,"failed restore published a native hit");
        run("parent");
        const auto before_cancel=wait_cold("cancel-read");
        auto cancelled_input=question; cancelled_input.context_cache.session_key="cancelled-child";
        failure.arm(true);
        std::atomic<bool> cancel{false};
        auto pending=engine.submit(engine.prepare(cancelled_input),response_options);
        auto cancelled=std::async(std::launch::async,[pending=std::move(pending),&cancel]() mutable {
            return pending.wait(nullptr,ninfer::CancellationView([&]{return cancel.load();}));
        });
        const bool entered=failure.wait_entered();
        cancel=true;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        failure.release();
        const auto result=cancelled.get();
        require(entered,"fork cancellation fixture did not enter a cold ReadFile");
        require(result.finish_reason==ninfer::FinishReason::Cancelled,
                "fork cancellation did not settle while reading cold state");
        const auto after_cancel=engine.memory_summary().kvmem;
        require(after_cancel.history_hits==before_cancel.history_hits &&
                after_cancel.disk_errors==before_cancel.disk_errors,
                "cancelled cold restore published a hit or an I/O error");
        require(!after_cancel.active_payload_reservation_bytes && !after_cancel.resident_pages &&
                !after_cancel.mtp_resident_pages && result.generated_token_ids.empty(),
                "cancelled cold restore generated output or leaked native resources");
        std::cout<<"FORK_CANCEL entered=1 cancelled=1 unpublished=1 resources_released=1"<<std::endl;
        run("parent"); run("child");
        auto edited=question;
        edited.context_cache.session_key="edited";
        edited.messages.front().parts.front().text="Different system identity; answer briefly.";
        const auto changed=engine.submit(engine.prepare(edited),request(4)).wait();
        require(!changed.reused_prompt_tokens,"different prefix incorrectly reused history");
        auto no_update=question; no_update.context_cache.session_key="no-update";
        no_update.context_cache.update_session_index=false;
        require(!engine.submit(engine.prepare(no_update),request(4)).wait().reused_prompt_tokens,
                "no-update request consumed named history");
        run("parent");
        std::cout<<"FORK_STORAGE observed_max_payload="<<observed_disk_payload
                 <<" quota="<<options.kvmem.disk_bytes<<std::endl;
        std::cout<<"CHECKPOINT_FORK cold_read_failure=1 cancel=1 parent_preserved=1 fresh_exact=1 PASS"<<std::endl;
    }
    require(std::filesystem::is_empty(options.kvmem.disk_path),"fork leaked spill file");
    std::filesystem::remove(options.kvmem.disk_path);
}
}
int main(int argc, char** argv) {
    try {
        if(argc==3 && std::string_view(argv[2])=="--fork") {checkpoint_fork(argv[1]);return 0;}
        require(argc == 2, "artifact");
        cold_rewrite_peer_progress();
        const auto promotion_path = std::filesystem::current_path() / "promotion-spill";
        require(std::filesystem::is_empty(promotion_path), "promotion leaked its spill file");
        std::filesystem::remove(promotion_path);
        ninfer::EngineOptions options;
        ninfer::test::kvmem_test_devices(options);
        options.artifact_path = argv[1];
        options.device_profile = "off";
        options.max_context = 8192;
        options.max_concurrency = 2;
        options.prefill_chunk = 128;
        options.kv_cache = ninfer::KvCacheStorage::Int8Group64;
        options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(1024);
        options.kvmem = {.selected_tokens = 384, .reserve_tokens = 128,
            .host_bytes = 512ULL << 20, .retained_sessions = 2, .verify_transfers = true};
        options.context_cache.enabled = false;
        std::vector<std::int32_t> reference;
        {
            ninfer::Engine engine(options);
            const auto result = engine.submit(engine.prepare(prompt(false)), request(256)).wait();
            reference.assign(result.generated_token_ids.begin(), result.generated_token_ids.end());
        }
        options.kvmem.host_bytes = 48ULL << 20;
        options.kvmem.disk_bytes = 1024ULL << 20;
        options.kvmem.disk_path = std::filesystem::temp_directory_path() /
            ("ninfer-ssd-failure-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        WriteFailure failure;
        {
            ninfer::Engine engine(options);
            Sink sink;
            auto first = sink.first.get_future();
            auto hot = engine.submit(engine.prepare(prompt(false)), request(256),
                ninfer::OutputConsumerMode::Streaming, {.live_timings = true});
            auto peer = std::async(std::launch::async, [hot = std::move(hot), &sink]() mutable { return hot.wait(&sink); });
            require(first.wait_for(std::chrono::seconds(60)) == std::future_status::ready, "hot peer never started");
            failure.arm();
            bool failed = false;
            try { (void)engine.submit(engine.prepare(prompt(true)), request(32)).wait(); }
            catch (const std::exception& error) {
                failed = true;
                std::cout << "EXPECTED_STORAGE_FAILURE " << error.what() << std::endl;
            }
            require(failed && failure.count() == 1, "cold lane did not receive the injected write failure");
            const auto survived = peer.get();
            require(std::vector<std::int32_t>(survived.generated_token_ids.begin(), survived.generated_token_ids.end()) == reference,
                    "storage failure changed the hot peer output");
            require(engine.is_available(), "one storage failure disabled the Engine");
            const auto recovered = engine.submit(engine.prepare(prompt(true)), request(32)).wait();
            require(recovered.generated_token_ids.size() == 32 && engine.is_available(), "retry after storage failure did not complete");
            const auto memory = engine.memory_summary().kvmem;
            require(memory.disk_errors == 1 && memory.disk_written_bytes && memory.host_payload_bytes <= options.kvmem.host_bytes &&
                    memory.disk_payload_bytes <= options.kvmem.disk_bytes && memory.resident_pages == 0,
                    "failure recovery leaked quota/pages or lost I/O accounting");
            std::cout << "SSD_FAILURE errors=" << memory.disk_errors << " peer_exact=1 retry=1" << std::endl;
        }
        {
            ninfer::Engine engine(options);
            Sink sink;
            auto first = sink.first.get_future();
            auto hot = engine.submit(engine.prepare(prompt(false)), request(512),
                ninfer::OutputConsumerMode::Streaming, {.live_timings = true});
            auto peer = std::async(std::launch::async, [hot = std::move(hot), &sink]() mutable { return hot.wait(&sink); });
            require(first.wait_for(std::chrono::seconds(60)) == std::future_status::ready, "cancellation peer never started");
            failure.arm(true);
            std::atomic<bool> cancel{false};
            auto cold = engine.submit(engine.prepare(prompt(true)), request(32));
            auto cancelled = std::async(std::launch::async, [cold = std::move(cold), &cancel]() mutable {
                return cold.wait(nullptr, ninfer::CancellationView([&] { return cancel.load(); }));
            });
            const bool blocked = failure.wait_entered();
            const auto before = sink.generated.load();
            cancel = true;
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
            while (sink.generated.load() < before + 8 && std::chrono::steady_clock::now() < deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            const auto advanced = sink.generated.load() - before;
            failure.release(); // always unblock the worker before any assertion/cleanup
            require(blocked && advanced >= 8, "cancelled slow storage job blocked the hot peer");
            require(cancelled.get().finish_reason == ninfer::FinishReason::Cancelled,
                    "cancelled storage lane did not settle as cancelled");
            require(peer.get().generated_token_ids.size() == 512 && engine.is_available(),
                    "storage cancellation damaged the peer or Engine");
            std::cout << "SSD_CANCEL peer_tokens_while_io_blocked=" << advanced << " PASS" << std::endl;
        }
        require(std::filesystem::is_empty(options.kvmem.disk_path), "failure path leaked its spill file");
        std::filesystem::remove(options.kvmem.disk_path);
        std::cout << "SSD_FAILURE PASS" << std::endl;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
#else
int main() { std::cout << "Windows import fault injection requires Windows; Linux OS failure coverage is in spill_io_test\n"; return 77; }
#endif
