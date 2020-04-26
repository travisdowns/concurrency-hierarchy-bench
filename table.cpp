/*
 * table.hpp
 *
 * Simple tabular output.
 */

#include "table.hpp"

#include <vector>
#include <sstream>
#include <algorithm>
#include <cassert>
#include <iomanip>
#include <memory>

namespace table {



    /** return a vector of sizes for each element */
std::vector<size_t> Row::getSizes() const {
    std::vector<size_t> sizes;
    for (const auto& e : elems_) {
        sizes.push_back(e.size());
    }
    return sizes;
}


std::string Row::justify(const ColInfo& cinfo, const std::string& e, size_t w) const {
    // left pad
    std::stringstream ss;
    ss << std::setw(w) << (cinfo.justify == ColInfo::LEFT ? std::left : std::right) << e;
    auto s = ss.str();
    assert(s.size() == w);
    return s;
}


/**
 * Get a reference to the ColInfo object for the given column, which lets you
 * set column-global info such as the justification.
 */
ColInfo& Table::colInfo(size_t col) {
    if (col >= colinfo_.size()) {
        colinfo_.resize(col + 1);
    }
    return colinfo_.at(col);
}

/* in the cost case, return a default ColInfo if it doesn't exist */
ColInfo Table::colInfo(size_t col) const {
    return col < colinfo_.size() ? colinfo_.at(col) : ColInfo{};
}

Row& Table::newRow() {
    rows_.push_back(Row{*this});
    return rows_.back();
}

/** return the current representation of the table as a string */
std::string Table::str() const {

    // calculate max row sizes
    std::vector<size_t> max_sizes;
    for (const auto& r : rows_) {
        std::vector<size_t> sizes = r.getSizes();
        for (size_t c = 0; c < sizes.size(); c++) {
            size_t row_size = sizes[c];
            if (c >= max_sizes.size()) {
                assert(max_sizes.size() == c);
                max_sizes.push_back(row_size);
            } else {
                max_sizes[c] = std::max(max_sizes[c], row_size);
            }
        }
    }

    std::stringstream ss;
    for (const auto& r : rows_) {
        r.str(ss, max_sizes);
        ss << "\n";
    }

    return ss.str();
}


void Row::str(std::ostream& os, const std::vector<size_t> sizes) const
{
    bool first = true;
    for (size_t c = 0; c < elems_.size(); c++) {
        const auto& e = elems_[c];
        assert(c < sizes.size());
        if (!first) os << table_->sep; // inter-cell padding
        first = false;
        os << justify(table_->colInfo(c), e, sizes[c]);
    }
}

std::string Table::csv_str() const {
    std::string out;
    for (const auto& r : rows_) {
        r.csv_str(out);
        out += '\n';
    }
    return out;
}

void Row::csv_str(std::string& out) const {
    bool first = true;
    for (auto& cell : elems_) {
        out += first ? "" : ",";
        out += cell;
        first = false;
    }
}

Row& Row::add_string(const std::string& s) {
    elems_.push_back(s);
    return *this;
}


template <typename T>
Row& add_inner(Row& r, const T& elem) {
    std::stringstream ss;
    ss << elem;
    return r.add_string(ss.str());
}

#define DEFINE_ADD(T) Row& Row::add(T t) { return add_inner(*this, t); }
SUPPORTED_TYPES_X(DEFINE_ADD)

}