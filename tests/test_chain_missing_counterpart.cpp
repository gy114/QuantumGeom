#include "Chain.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

Chain make_chain_with_missing_positive_counterpart() {
    const std::vector<int> translations{-2, -1, 0, 2};
    std::vector<arma::cx_mat> hamiltonians;
    std::vector<arma::cx_mat> overlaps;
    for (size_t i = 0; i < translations.size(); ++i) {
        hamiltonians.emplace_back(1, 1, arma::fill::ones);
        overlaps.emplace_back(1, 1, arma::fill::ones);
    }
    return Chain(hamiltonians, translations, overlaps);
}

template <typename Function>
void expect_runtime_error(Function&& function, const std::string& expected_message) {
    try {
        function();
    } catch (const std::runtime_error& error) {
        if (error.what() != expected_message) {
            throw std::runtime_error(
                "Unexpected error message: " + std::string(error.what()));
        }
        return;
    } catch (const std::exception& error) {
        throw std::runtime_error(
            "Expected std::runtime_error, received a different exception: " +
            std::string(error.what()));
    }
    throw std::runtime_error("Expected std::runtime_error, but no exception was thrown");
}

}  // namespace

int main() {
    try {
        const Chain chain = make_chain_with_missing_positive_counterpart();
        expect_runtime_error(
            [&] { chain.generateLocalHamiltonian(0.1); },
            "Missing H(-R) for R = -1");
        expect_runtime_error(
            [&] { chain.generateLocalOverlap(0.1); },
            "Missing S(-R) for R = -1");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }

    std::cout << "Missing counterparts produce descriptive runtime errors\n";
    return 0;
}
