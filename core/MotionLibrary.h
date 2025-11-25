#ifndef __MASS_MOTION_LIBRARY_H__
#define __MASS_MOTION_LIBRARY_H__

#include <vector>
#include <string>
#include <map>
#include <random>
#include "dart/dart.hpp"

namespace MASS
{

class BVH;

/**
 * MotionEntry: Represents a single motion clip
 */
struct MotionEntry
{
    std::string filepath;
    std::string name;
    bool cyclic;
    BVH* bvh;
    double duration;
    
    MotionEntry() : filepath(""), name(""), cyclic(true), bvh(nullptr), duration(0.0) {}
};

/**
 * MotionLibrary: Manages multiple BVH files for multimodal motion imitation
 * 
 * This class provides:
 * - Loading multiple BVH files from a motion list
 * - Random motion selection for training diversity
 * - Motion switching during episode resets
 * 
 * Usage:
 *   1. Create MotionLibrary with skeleton
 *   2. Call LoadMotionList() with path to motion_list.txt
 *   3. Call SelectRandomMotion() on each reset
 *   4. GetCurrentBVH() returns the active motion
 */
class MotionLibrary
{
public:
    MotionLibrary(const dart::dynamics::SkeletonPtr& skel, 
                  const std::map<std::string, std::string>& bvh_map);
    ~MotionLibrary();
    
    /**
     * Load motions from a list file
     * Format per line: <filepath> <cyclic: true/false>
     * Lines starting with # are comments
     */
    bool LoadMotionList(const std::string& list_path);
    
    /**
     * Load a single motion and add to library
     */
    bool AddMotion(const std::string& filepath, bool cyclic = true);
    
    /**
     * Select a random motion from the library
     * Returns the index of selected motion
     */
    int SelectRandomMotion();
    
    /**
     * Select a specific motion by index
     */
    bool SelectMotion(int index);
    
    /**
     * Select a specific motion by name
     */
    bool SelectMotion(const std::string& name);
    
    // Getters
    BVH* GetCurrentBVH() const { return mCurrentBVH; }
    int GetCurrentIndex() const { return mCurrentIndex; }
    const std::string& GetCurrentName() const;
    int GetNumMotions() const { return static_cast<int>(mMotions.size()); }
    const std::vector<MotionEntry>& GetMotions() const { return mMotions; }
    
    /**
     * Get motion weights for sampling (can be set based on difficulty, duration, etc.)
     */
    const std::vector<double>& GetMotionWeights() const { return mMotionWeights; }
    void SetMotionWeights(const std::vector<double>& weights);
    
    /**
     * Get statistics about motion usage during training
     */
    const std::vector<int>& GetMotionCounts() const { return mMotionCounts; }
    void ResetMotionCounts();
    
private:
    dart::dynamics::SkeletonPtr mSkeleton;
    std::map<std::string, std::string> mBVHMap;
    
    std::vector<MotionEntry> mMotions;
    std::vector<double> mMotionWeights;
    std::vector<int> mMotionCounts;
    
    BVH* mCurrentBVH;
    int mCurrentIndex;
    
    std::mt19937 mRNG;
    
    static std::string empty_string;
};

} // namespace MASS

#endif // __MASS_MOTION_LIBRARY_H__