// How tensor-parallel ranks divide a dimension (split_shares) and the work by speed and room
// (fill_shares). The expected splits are worked out by hand from the stated rule: every inner end is
// the allowed position nearest its weight's target, the lower one on a tie, leaving each later share
// a position.

#include "core/tensor_split.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* label) {
    if (condition) { return; }
    std::cerr << "expectation failed: " << label << '\n';
    ++failures;
}

template <class Fn>
void check_throws(Fn&& fn, const char* label) {
    try {
        fn();
    } catch (const std::invalid_argument&) {
        return;
    }
    std::cerr << "expectation failed: " << label << " (nothing was thrown)\n";
    ++failures;
}

using Ends = std::vector<std::uint64_t>;

Ends split(std::uint64_t total, std::uint64_t unit, std::vector<double> weights) {
    return ninfer::split_shares(total, ninfer::unit_positions(total, unit), weights);
}

bool near(const std::vector<double>& actual, const std::vector<double>& expected) {
    if (actual.size() != expected.size()) { return false; }
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (std::abs(actual[i] - expected[i]) > 1e-12) { return false; }
    }
    return true;
}

void split_cases() {
    // Equal weights over evenly spaced positions: equal shares (16 GDN key heads).
    check(split(16, 1, {1, 1}) == Ends{0, 8, 16}, "two equal ranks");
    check(split(16, 1, {1, 1, 1, 1}) == Ends{0, 4, 8, 12, 16}, "four equal ranks");
    // 3:1 puts the end at 12.
    check(split(16, 1, {3, 1}) == Ends{0, 12, 16}, "3:1");
    // Three equal ranks of 16: targets 5.33 and 10.67.
    check(split(16, 1, {1, 1, 1}) == Ends{0, 5, 11, 16}, "three equal ranks");
    // Only even ends (two heads per unit): 5.33 -> 6, 10.67 -> 10.
    check(split(16, 2, {1, 1, 1}) == Ends{0, 6, 10, 16}, "three ranks, two-head units");
    // A tie: units of three over 15 put the target 7.5 midway between 6 and 9; the lower wins.
    check(split(15, 3, {1, 1}) == Ends{0, 6, 15}, "a tie takes the lower end");
    // An overwhelming weight still leaves the last rank one position.
    check(split(16, 1, {1000, 1}) == Ends{0, 15, 16}, "the last share keeps one unit");
    check(split(16, 1, {1, 1000}) == Ends{0, 1, 16}, "the first share keeps one unit");
    check(split(16, 4, {1, 1000, 1000, 1000}) == Ends{0, 4, 8, 12, 16},
          "later ranks keep their positions");
    // An FFN of 17408 rows in 256-row blocks at an RTX 3090 : RTX 3060 bandwidth ratio: the target
    // 12572.4 lies nearest the block end 12544.
    check(split(17408, 256, {936, 360}) == Ends{0, 12544, 17408}, "FFN rows by bandwidth");
    // Explicit positions (a gathered projection's whole blocks).
    const Ends allowed{3, 4, 9, 10};
    check(ninfer::split_shares(12, allowed, std::vector<double>{1, 1}) == Ends{0, 4, 12},
          "explicit positions: 6 is nearer 4 than 9");
    check(ninfer::split_shares(12, Ends{}, std::vector<double>{1}) == Ends{0, 12}, "one rank");

    check_throws([] { (void)split(4, 2, {1, 1, 1}); }, "too few positions for the ranks");
    check_throws([] { (void)split(16, 1, {1, 0}); }, "a zero weight");
    check_throws([] { (void)split(16, 1, {1, -2}); }, "a negative weight");
    check_throws([] { (void)split(16, 1, {}); }, "no ranks");
    check_throws([] { (void)ninfer::split_shares(8, Ends{4, 4}, std::vector<double>{1, 1, 1}); },
                 "repeated positions");
    check_throws([] { (void)ninfer::split_shares(8, Ends{8}, std::vector<double>{1, 1}); },
                 "a position at the end");
}

void fill_cases() {
    // Room everywhere: in proportion to speed.
    check(near(ninfer::fill_shares(std::vector<double>{3, 1}, std::vector<double>{9, 9}),
               {0.75, 0.25}),
          "proportional to speed");
    // A full rank's excess goes to the other.
    check(near(ninfer::fill_shares(std::vector<double>{1, 1}, std::vector<double>{0.3, 2}),
               {0.3, 0.7}),
          "capped rank");
    // The excess goes to the open ranks in proportion to their speed.
    check(near(ninfer::fill_shares(std::vector<double>{2, 1, 1}, std::vector<double>{0.25, 1, 1}),
               {0.25, 0.375, 0.375}),
          "excess shared by speed");
    // Capping one rank can overflow another, which is capped in turn.
    check(near(ninfer::fill_shares(std::vector<double>{1, 1, 1}, std::vector<double>{0.1, 0.2, 1}),
               {0.1, 0.2, 0.7}),
          "cascading caps");
    check(near(ninfer::fill_shares(std::vector<double>{4, 1, 1}, std::vector<double>{0.3, 0.6, 0.3}),
               {0.3, 0.4, 0.3}),
          "a cap reached only after redistribution");
    // Not enough room anywhere: speed alone.
    check(near(ninfer::fill_shares(std::vector<double>{1, 3}, std::vector<double>{0.2, 0.2}),
               {0.25, 0.75}),
          "no fit follows speed");
    // Not enough room, but the work shrinks to fit (automatic KV): in proportion to the room, so
    // both ranks outgrow theirs by the same factor. Without room on one rank, speed again.
    check(near(ninfer::fill_shares(std::vector<double>{1, 1}, std::vector<double>{0.3, 0.6}, true),
               {1.0 / 3.0, 2.0 / 3.0}),
          "a shrinking fit follows room");
    check(near(ninfer::fill_shares(std::vector<double>{1, 1}, std::vector<double>{0.4, 0.8}, true),
               {0.4, 0.6}),
          "a shrinking fit that fits caps as usual");
    check(near(ninfer::fill_shares(std::vector<double>{1, 3}, std::vector<double>{0, 0.5}, true),
               {0.25, 0.75}),
          "a rank without room falls back to speed");
    check_throws([] { (void)ninfer::fill_shares(std::vector<double>{1}, std::vector<double>{}); },
                 "a capacity per rank");
    check_throws(
        [] { (void)ninfer::fill_shares(std::vector<double>{0, 1}, std::vector<double>{1, 1}); },
        "a zero speed");
}

} // namespace

int run_tensor_split_test() {
    try {
        split_cases();
        fill_cases();
    } catch (const std::exception& error) {
        std::cerr << "unexpected exception: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) { return 1; }
    std::cout << "tensor split tests passed\n";
    return 0;
}
