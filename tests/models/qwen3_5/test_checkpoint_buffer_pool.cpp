#include "models/qwen3_5/program/storage/checkpoint_buffer_pool.h"
#include <cstring>
#include <stdexcept>

#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (false)
int main() {
    using namespace ninfer::models::qwen3_5::detail;
    auto pool = std::make_shared<KvmemCheckpointBufferPool>();
    auto first = pool->acquire(4096);
    auto other = pool->acquire(4096);
    CHECK(first->data() != other->data());
    std::memset(first->data(), 17, first->size());
    std::memset(other->data(), 23, other->size());
    auto alias = first;
    const auto address = first->data();
    first.reset();
    CHECK(pool->idle_bytes() == 0); // An aliased checkpoint is still live.
    alias.reset();
    CHECK(pool->idle_bytes() == 4096);
    first = pool->acquire(4096);
    CHECK(first->data() == address && pool->idle_bytes() == 0);
    std::memset(first->data(), 29, first->size());
    for (std::size_t i = 0; i < other->size(); ++i)
        CHECK(static_cast<unsigned char*>(other->data())[i] == 23);
    first.reset(); other.reset();
    CHECK(pool->idle_bytes() == 8192);
    auto extra = pool->acquire(8192);
    extra.reset();
    CHECK(pool->idle_bytes() == 8192); // A third release cannot grow the cache.
    auto last = pool->acquire(4096);
    pool.reset(); // Outstanding leases own the recycler's lifetime.
    std::memset(last->data(), 31, last->size());
    last.reset();
    return 0;
}
