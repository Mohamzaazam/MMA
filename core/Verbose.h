#ifndef __MASS_VERBOSE_H__
#define __MASS_VERBOSE_H__

namespace MASS {

// Global verbosity flag for controlling console output
// Set to false to suppress BVH parsing and skeleton loading messages
extern bool gVerbose;

// Helper macro for verbose output
#define MASS_LOG_INFO(...) if (MASS::gVerbose) { std::cout << __VA_ARGS__; }

} // namespace MASS

#endif // __MASS_VERBOSE_H__
