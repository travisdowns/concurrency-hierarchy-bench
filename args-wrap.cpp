#include "args-wrap.hpp"
#include "args.hxx"

namespace argsw {

std::array<char,0> empty_short;

ArgumentParser::ArgumentParser(const std::string &description_, const std::string &epilog_) {
    delegate = new args::ArgumentParser(description_, epilog_);
}

ArgumentParser::~ArgumentParser() {
    delete delegate;
}

bool ArgumentParser::ParseCLI(const int argc, const char * const * argv, string_consumer* help, string_consumer* parse_error) {
    try {
        return delegate->ParseCLI(argc, argv);
    } catch (const args::Help& e) {
        if (help) {
            help(delegate->Help());
        } else {
            throw;
        }
    } catch (const args::ParseError& e) {
        if (parse_error) {
            parse_error(e.what());
        } else {
            throw;
        }
    }
    return false;
}

std::string ArgumentParser::Help() const {
    return delegate->Help();
}

HelpFlag::HelpFlag(ArgumentParser& group, const std::string &name_, const std::string &help_, std::initializer_list<std::string> matcher_) {
    delegate = new args::HelpFlag(*group.delegate, name_, help_, {empty_short, matcher_});
}

HelpFlag::~HelpFlag() {
    delete delegate;
}

Flag::Flag(ArgumentParser& group, const std::string &name_, const std::string &help_, std::initializer_list<std::string> matcher_) {
    base = delegate = new args::Flag(*group.delegate, name_, help_, {empty_short, matcher_});
}

Flag::~Flag() {
    delete delegate;
}

template <typename T>
ValueFlag<T>::ValueFlag(ArgumentParser& group, const std::string &name_, const std::string &help_, std::initializer_list<std::string> matcher_,
        const T &defaultValue_) {
    base = delegate = new args::ValueFlag<T,args::ValueReader>(*group.delegate, name_, help_, {empty_short, matcher_}, defaultValue_);
}

template <typename T>
ValueFlag<T>::~ValueFlag() {
    delete delegate;
}

template <typename T>
T& ValueFlag<T>::Get() noexcept {
    return delegate->Get();
}

Base::operator bool() const noexcept
{
    return base->Matched();
}

#define VF_TYPES_X(fn) \
    fn(int) \
    fn(size_t) \
    fn(double) \
    fn(std::string)

#define EXPLICIT_VF(type) template class ValueFlag<type>;

// explicitly instantiate the ValueFlag specializations
VF_TYPES_X(EXPLICIT_VF);

};
