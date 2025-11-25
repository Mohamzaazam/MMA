#include "MotionLibrary.h"
#include "BVH.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

namespace MASS
{

std::string MotionLibrary::empty_string = "";

MotionLibrary::MotionLibrary(const dart::dynamics::SkeletonPtr& skel,
                             const std::map<std::string, std::string>& bvh_map)
    : mSkeleton(skel)
    , mBVHMap(bvh_map)
    , mCurrentBVH(nullptr)
    , mCurrentIndex(-1)
{
    // Seed RNG with random device
    std::random_device rd;
    mRNG.seed(rd());
}

MotionLibrary::~MotionLibrary()
{
    // Clean up BVH objects
    for (auto& entry : mMotions)
    {
        if (entry.bvh != nullptr)
        {
            delete entry.bvh;
            entry.bvh = nullptr;
        }
    }
}

bool MotionLibrary::LoadMotionList(const std::string& list_path)
{
    std::ifstream ifs(list_path);
    if (!ifs.is_open())
    {
        std::cerr << "MotionLibrary: Cannot open motion list: " << list_path << std::endl;
        return false;
    }
    
    std::string line;
    int loaded_count = 0;
    int failed_count = 0;
    
    // Get base directory from list_path for relative paths
    fs::path base_dir = fs::path(list_path).parent_path().parent_path(); // Go up from data/
    
    while (std::getline(ifs, line))
    {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#')
            continue;
        
        std::istringstream iss(line);
        std::string filepath;
        std::string cyclic_str;
        
        iss >> filepath >> cyclic_str;
        
        if (filepath.empty())
            continue;
        
        // Handle relative paths
        fs::path motion_path(filepath);
        if (motion_path.is_relative())
        {
            motion_path = base_dir / motion_path;
        }
        
        bool cyclic = (cyclic_str == "true" || cyclic_str == "True" || cyclic_str == "1");
        
        if (AddMotion(motion_path.string(), cyclic))
        {
            loaded_count++;
        }
        else
        {
            failed_count++;
        }
    }
    
    ifs.close();
    
    std::cout << "MotionLibrary: Loaded " << loaded_count << " motions";
    if (failed_count > 0)
    {
        std::cout << " (" << failed_count << " failed)";
    }
    std::cout << std::endl;
    
    // Initialize weights uniformly
    if (!mMotions.empty())
    {
        mMotionWeights.resize(mMotions.size(), 1.0);
        mMotionCounts.resize(mMotions.size(), 0);
        
        // Select first motion by default
        SelectMotion(0);
    }
    
    return loaded_count > 0;
}

bool MotionLibrary::AddMotion(const std::string& filepath, bool cyclic)
{
    // Check file exists
    if (!fs::exists(filepath))
    {
        std::cerr << "MotionLibrary: File not found: " << filepath << std::endl;
        return false;
    }
    
    // Create BVH object
    BVH* bvh = new BVH(mSkeleton, mBVHMap);
    
    try
    {
        bvh->Parse(filepath, cyclic);
    }
    catch (const std::exception& e)
    {
        std::cerr << "MotionLibrary: Failed to parse " << filepath << ": " << e.what() << std::endl;
        delete bvh;
        return false;
    }
    
    // Create motion entry
    MotionEntry entry;
    entry.filepath = filepath;
    entry.name = fs::path(filepath).stem().string(); // filename without extension
    entry.cyclic = cyclic;
    entry.bvh = bvh;
    entry.duration = bvh->GetMaxTime();
    
    mMotions.push_back(entry);
    
    std::cout << "  + Loaded: " << entry.name 
              << " (duration: " << entry.duration << "s, "
              << (cyclic ? "cyclic" : "non-cyclic") << ")" << std::endl;
    
    return true;
}

int MotionLibrary::SelectRandomMotion()
{
    if (mMotions.empty())
    {
        std::cerr << "MotionLibrary: No motions loaded!" << std::endl;
        return -1;
    }
    
    // Weighted random selection
    std::vector<double> cumulative_weights(mMotionWeights.size());
    cumulative_weights[0] = mMotionWeights[0];
    for (size_t i = 1; i < mMotionWeights.size(); i++)
    {
        cumulative_weights[i] = cumulative_weights[i-1] + mMotionWeights[i];
    }
    
    std::uniform_real_distribution<double> dist(0.0, cumulative_weights.back());
    double r = dist(mRNG);
    
    int selected = 0;
    for (size_t i = 0; i < cumulative_weights.size(); i++)
    {
        if (r <= cumulative_weights[i])
        {
            selected = static_cast<int>(i);
            break;
        }
    }
    
    SelectMotion(selected);
    return selected;
}

bool MotionLibrary::SelectMotion(int index)
{
    if (index < 0 || index >= static_cast<int>(mMotions.size()))
    {
        std::cerr << "MotionLibrary: Invalid motion index: " << index << std::endl;
        return false;
    }
    
    mCurrentIndex = index;
    mCurrentBVH = mMotions[index].bvh;
    mMotionCounts[index]++;
    
    return true;
}

bool MotionLibrary::SelectMotion(const std::string& name)
{
    for (size_t i = 0; i < mMotions.size(); i++)
    {
        if (mMotions[i].name == name)
        {
            return SelectMotion(static_cast<int>(i));
        }
    }
    
    std::cerr << "MotionLibrary: Motion not found: " << name << std::endl;
    return false;
}

const std::string& MotionLibrary::GetCurrentName() const
{
    if (mCurrentIndex >= 0 && mCurrentIndex < static_cast<int>(mMotions.size()))
    {
        return mMotions[mCurrentIndex].name;
    }
    return empty_string;
}

void MotionLibrary::SetMotionWeights(const std::vector<double>& weights)
{
    if (weights.size() != mMotions.size())
    {
        std::cerr << "MotionLibrary: Weight vector size mismatch" << std::endl;
        return;
    }
    mMotionWeights = weights;
}

void MotionLibrary::ResetMotionCounts()
{
    std::fill(mMotionCounts.begin(), mMotionCounts.end(), 0);
}

} // namespace MASS