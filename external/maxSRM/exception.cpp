#include <exception>
#include <string>

/* Simple class for throwing exceptions */
class exception : public std::exception {
public:
    exception(std::string message) { this->message = message; }
    virtual ~exception() throw() {}
    virtual const char* what() const throw() { return message.c_str(); }
private:
    std::string message;
};