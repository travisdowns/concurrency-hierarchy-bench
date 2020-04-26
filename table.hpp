/*
 * table.hpp
 *
 * Simple tabular output.
 */

#ifndef TABLE_HPP_
#define TABLE_HPP_

#include <vector>
#include <cassert>
#include <string>
#include <stdexcept>

#ifdef TABLE_ARBITRARY_ELEMS
#include <iomanip>
#include <sstream>
#endif

#define SUPPORTED_TYPES_X(fn) \
    fn(int) \
    fn(long) \
    fn(long long) \
    fn(unsigned int) \
    fn(unsigned long) \
    fn(unsigned long long) \
    fn(double) \
    fn(const char *) \
    fn(const std::string& )

namespace table {

class Table;

struct ColInfo {
    enum Justify { LEFT, RIGHT } justify;
    ColInfo() : justify(LEFT) {}
};

/*
 * Given a printf-style format and args, return the formatted string as a std::string.
 *
 * See https://stackoverflow.com/a/26221725/149138.
 */
template<typename ... Args>
std::string string_format(const std::string& format, Args ... args) {
    int size = snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    if (size < 0) {
        throw std::runtime_error("failed while formatting: " + format);
    }
    char* buf = new char[size + 1];
    snprintf( buf, size + 1, format.c_str(), args ... );
    assert(buf[size] == '\0');
    std::string ret(buf); // We don't want the '\0' inside
    delete [] buf;
    return ret;
}

class Row {
    friend Table;
    using row_t = std::vector<std::string>;

    const Table* table_;
    row_t elems_;

    Row(const Table& table) : table_(&table) {}

    /** return a vector of sizes for each element */
    std::vector<size_t> getSizes() const;

    void str(std::ostream& os, const std::vector<size_t> sizes) const;
    void csv_str(std::string& out) const;

    std::string justify(const ColInfo& cinfo, const std::string& e, size_t w) const;

public:

#ifdef TABLE_ARBITRARY_ELEMENTS
    /** add a cell to this row with the given element, returns a reference to this row */
    template <typename T>
    Row& add(const T& elem) {
        std::stringstream ss;
        ss << elem;
        elems_.push_back(ss.str());
        return *this;
    }
#endif

#define DECLARE_ADD(T) Row& add(T);
    SUPPORTED_TYPES_X(DECLARE_ADD)

    Row& add_string(const std::string& s);

    /**
     * Add a formatted cell to this row with the given element.
     * The format is a printf-style format string and any additional arguments are the format arguments.
     * Returns a reference to this row.
     */
    template <typename ... Args>
    Row& addf(const char* format, Args ... args) {
        elems_.push_back(string_format(format, args...));
        return *this;
    }
};

class Table {
    friend Row;
    using table_t   = std::vector<Row>;
    using colinfo_t = std::vector<ColInfo>;

    table_t rows_;
    colinfo_t colinfo_;
    std::string sep;

public:

    Table() : sep(" ") {}

    /**
     * Get a reference to the ColInfo object for the given column, which lets you
     * set column-global info such as the justification.
     */
    ColInfo& colInfo(size_t col);

    /* in the cost case, return a default ColInfo if it doesn't exist */
    ColInfo colInfo(size_t col) const;

    Row& newRow();

    /** return a representation of the table as a human readable, column-aligned string */
    std::string str() const;

    /** return the table as csv without padding */
    std::string csv_str() const;

    void setColColumnSeparator(std::string s) {
        sep = s;
    }

};

}


#endif /* TABLE_HPP_ */
