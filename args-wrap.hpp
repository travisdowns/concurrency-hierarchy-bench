#include <string>

namespace args {
class ArgumentParser;
class Group;
class Matcher;
class HelpFlag;
class Flag;
template <typename T, typename VR>
class ValueFlag;
struct ValueReader;
class Base;
}

namespace argsw {

class ArgumentParser;

class Base {
protected:
    args::Base* base;
public:
    operator bool() const noexcept;            
};

class HelpFlag {
    args::HelpFlag* delegate;

public:
    HelpFlag(ArgumentParser& group, const std::string &name_, const std::string &help_, std::initializer_list<std::string> matcher_);
    ~HelpFlag();
};

class Flag : public Base {
    args::Flag* delegate;

public:
    Flag(ArgumentParser& group, const std::string &name_, const std::string &help_, std::initializer_list<std::string> matcher_);
    ~Flag();
};

template <typename T>
class ValueFlag : public Base {
    args::ValueFlag<T, args::ValueReader>* delegate;

public:
    ValueFlag(ArgumentParser& group, const std::string &name_, const std::string &help_, std::initializer_list<std::string> matcher_,
            const T &defaultValue_ = T());
    ~ValueFlag();

    T &Get() noexcept;
};

using string_consumer = void(const std::string&);

class ArgumentParser {
    args::ArgumentParser* delegate;

    friend HelpFlag;
    friend Flag;
    template <class U>
    friend class ValueFlag;
public:
    ArgumentParser(const std::string& description_, const std::string& epilog_ = std::string());
    ~ArgumentParser();

    bool ParseCLI(const int argc, const char * const * argv, string_consumer* help = nullptr, string_consumer* parse_error = nullptr);
    std::string Help() const;

};

};  // namespace argsw