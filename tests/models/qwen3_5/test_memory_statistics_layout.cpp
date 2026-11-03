#include "models/qwen3_5/execution/memory_statistics.h"
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <type_traits>

using namespace ninfer;
using namespace ninfer::models::qwen3_5::execution;

template<class Query>
void check_values(bool pipeline, bool include_query, std::uint32_t accepted) {
    MemoryStatistics stats;
    stats.layers=2; stats.key_width=8; stats.query_width=16;
    stats.columns=3; stats.mean_query=std::is_same_v<Query,float>;
    stats.query_columns=stats.mean_query ? 1 : 4;
    stats.ranks.resize(3); // The last rank has no attention heads.
    std::array<std::vector<float>,2> keys;
    std::array<std::vector<Query>,2> queries;
    for(int rank=0;rank<2;++rank) {
        auto& shard=stats.ranks[rank];
        const int width=pipeline ? 8 : rank==0 ? 6 : 2;
        const int layers=pipeline ? 1 : 2;
        shard.first_layer=pipeline ? rank : 0;
        shard.key_offset=!pipeline && rank==1 ? 6 : 0;
        shard.query_offset=shard.key_offset*2;
        keys[rank].resize(size_t(width)*3*layers);
        queries[rank].resize(size_t(width*2)*stats.query_columns*layers);
        for(int l=0;l<layers;++l) {
            // Each feature carries its global identity, independent of the
            // host-copy plan, so a wrong layer/row/head offset is observable.
            for(int row=0;row<3;++row) for(int feature=0;feature<width;++feature)
                keys[rank][(size_t(l)*3+row)*width+feature]=
                    float(1000*(shard.first_layer+l)+100*row+shard.key_offset+feature+1);
            for(unsigned row=0;row<stats.query_columns;++row)
                for(int feature=0;feature<width*2;++feature)
                    queries[rank][(size_t(l)*stats.query_columns+row)*width*2+feature]=
                        Query(1000*(shard.first_layer+l)+100*row+shard.query_offset+feature+1);
        }
        shard.key_sums=Tensor(keys[rank].data(),DType::FP32,{width,3,layers});
        shard.query_values=Tensor(queries[rank].data(),stats.mean_query ? DType::FP32 : DType::BF16,
                                  {width*2,int(stats.query_columns),layers});
    }
    stats.validate_layout();
    constexpr size_t guard=32;
    std::vector<std::byte> actual(2*guard+stats.key_bytes()+stats.query_bytes(),std::byte{0xcd});
    auto expected=actual;
    for(unsigned layer=0;layer<2;++layer) {
        for(unsigned row=0;row<3;++row) for(unsigned feature=0;feature<8;++feature) {
            const float value=float(1000*layer+100*row+feature+1);
            const size_t at=guard+((size_t(layer)*3+row)*8+feature)*sizeof(value);
            std::memcpy(expected.data()+at,&value,sizeof(value));
        }
        if(include_query) for(unsigned row=0;row<(stats.mean_query ? 1 : accepted);++row)
            for(unsigned feature=0;feature<16;++feature) {
                const Query value=Query(1000*layer+100*row+feature+1);
                const size_t at=guard+stats.key_bytes()+
                    ((size_t(layer)*stats.query_columns+row)*16+feature)*sizeof(value);
                std::memcpy(expected.data()+at,&value,sizeof(value));
            }
    }
    stats.for_each_host_copy(include_query,accepted,[&](const MemoryStatisticsCopy& copy) {
        if(copy.rank>=2) throw std::runtime_error("empty rank scheduled a copy");
        if(copy.destination_offset+(copy.rows-1)*copy.destination_pitch+copy.width_bytes>
           stats.key_bytes()+stats.query_bytes())
            throw std::runtime_error("statistics copy exceeds host allocation");
        const auto* source=static_cast<const std::byte*>(copy.source);
        for(size_t row=0;row<copy.rows;++row)
            std::memcpy(actual.data()+guard+copy.destination_offset+row*copy.destination_pitch,
                        source+row*copy.width_bytes,copy.width_bytes);
    });
    if(actual!=expected)
        throw std::runtime_error("global layer/head order or accepted Query prefix differs from oracle");
    if(!stats.mean_query) for(const auto invalid:{0U,stats.query_columns+1}) {
        bool rejected=false, copied=false;
        try { stats.for_each_host_copy(true,invalid,[&](const auto&) {copied=true;}); }
        catch(const std::logic_error&) {rejected=true;}
        if(!rejected || copied) throw std::runtime_error("invalid Query prefix started a copy");
    }
}

int main() {
    // Shapes stand for 4 KV heads / 16 Query heads at head dimension 256.
    // The two ranks hold 3:1 heads; rank 2 deliberately has no attention.
    float sentinel = 0;
    MemoryStatistics stats;
    stats.layers = 2; stats.key_width = 1024; stats.query_width = 4096;
    stats.columns = 3; stats.query_columns = 7;
    stats.ranks.resize(3);
    for (int rank = 0; rank < 2; ++rank) {
        auto& shard = stats.ranks[rank];
        const int width = rank ? 256 : 768;
        shard.key_offset = rank ? 768 : 0;
        shard.query_offset = shard.key_offset * 4;
        shard.key_sums = Tensor(&sentinel, DType::FP32, {width, 3, 2});
        shard.query_values = Tensor(&sentinel, DType::BF16, {width * 4, 7, 2});
    }
    stats.validate_layout();
    for (int fault = 0; fault < 8; ++fault) {
        auto bad = stats;
        if (fault == 0) bad.ranks[1].key_offset = 512;
        if (fault == 1) bad.ranks[1].query_offset = 0;
        if (fault == 2) bad.ranks.pop_back(), bad.ranks.pop_back();
        if (fault == 3) bad.ranks[1].first_layer = 1;
        if (fault == 4) bad.ranks[0].key_sums.ne[1] = 2;
        if (fault == 5) bad.ranks[0].query_values.dtype = DType::FP32;
        if (fault == 6) bad.ranks[0].key_sums.nb[1] += sizeof(float);
        if (fault == 7) bad.ranks[0].query_values.ne[3] = 2;
        bool rejected = false;
        try { bad.validate_layout(); } catch (const std::logic_error&) { rejected = true; }
        if (!rejected) throw std::runtime_error("invalid statistics shard accepted");
    }
    // Layer pipeline is also a complete partition, with no head offsets.
    for (int rank = 0; rank < 2; ++rank) {
        auto& shard = stats.ranks[rank];
        shard.first_layer = rank; shard.key_offset = shard.query_offset = 0;
        shard.key_sums = Tensor(&sentinel, DType::FP32, {1024, 3, 1});
        shard.query_values = Tensor(&sentinel, DType::BF16, {4096, 7, 1});
    }
    stats.validate_layout();
    for(const bool pipeline:{false,true}) {
        check_values<std::uint16_t>(pipeline,true,3); // One unaccepted tail column stays untouched.
        check_values<std::uint16_t>(pipeline,true,4);
        check_values<std::uint16_t>(pipeline,false,0); // Mean-K capture without Query consumers.
        check_values<float>(pipeline,true,0);
        check_values<float>(pipeline,false,0);
    }
    std::cout << "OK: unequal heads, empty attention rank, layer pipeline, Query/MeanQ values, "
                 "accepted prefixes and invalid coverage\n";
}
