#ifndef __MASS_BVH_H__
#define __MASS_BVH_H__

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <string>
#include <fstream>
#include <vector>
#include <map>
#include <utility>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include "dart/dart.hpp"

namespace MASS
{

//------------------------------------------------------------------------------
// Rotation Helper Functions
//------------------------------------------------------------------------------

Eigen::Matrix3d R_x(double x);
Eigen::Matrix3d R_y(double y);
Eigen::Matrix3d R_z(double z);

//------------------------------------------------------------------------------
// Parser - Character-level parser with error tracking
//------------------------------------------------------------------------------

class Parser
{
public:
    static constexpr int ERR_MAX = 512;
    
    std::string filename;
    int offset;
    std::string data;
    int row;
    int col;
    char err[ERR_MAX];
    
    void Init(const std::string& fname, const std::string& content);
    char Peek() const;
    char PeekForward(int steps) const;
    bool Match(char c) const;
    bool OneOf(const char* chars) const;
    bool StartsWithCaseless(const char* prefix) const;
    void Inc();
    void Advance(int num);
    const char* CharName(char c);
    void SetError(const char* fmt, ...);
};

//------------------------------------------------------------------------------
// Channel Types
//------------------------------------------------------------------------------

enum Channel
{
    CHANNEL_X_POSITION = 0,
    CHANNEL_Y_POSITION = 1,
    CHANNEL_Z_POSITION = 2,
    CHANNEL_X_ROTATION = 3,
    CHANNEL_Y_ROTATION = 4,
    CHANNEL_Z_ROTATION = 5,
    CHANNEL_MAX = 6
};

//------------------------------------------------------------------------------
// BVHNode - Single joint in the BVH hierarchy
//------------------------------------------------------------------------------

class BVHNode
{
public:
    BVHNode(const std::string& name, BVHNode* parent);
    
    void SetChannel(int c_offset, const std::vector<Channel>& channels);
    void Set(const Eigen::VectorXd& m_t);
    void Set(const Eigen::Matrix3d& R_t);
    Eigen::Matrix3d Get() const;
    
    void AddChild(BVHNode* child);
    BVHNode* GetNode(const std::string& name);
    
    const std::string& GetName() const { return mName; }
    BVHNode* GetParent() const { return mParent; }
    const Eigen::Vector3d& GetOffset() const { return mOffset; }
    void SetOffset(const Eigen::Vector3d& offset) { mOffset = offset; }
    bool IsEndSite() const { return mEndSite; }
    void SetEndSite(bool endSite) { mEndSite = endSite; }
    int GetChannelOffset() const { return mChannelOffset; }
    int GetNumChannels() const { return mNumChannels; }

private:
    BVHNode* mParent;
    std::vector<BVHNode*> mChildren;
    
    Eigen::Matrix3d mR;
    std::string mName;
    Eigen::Vector3d mOffset;
    bool mEndSite;
    
    int mChannelOffset;
    int mNumChannels;
    std::vector<Channel> mChannels;  // Actual rotation order from file
};

//------------------------------------------------------------------------------
// BVH - Main BVH data structure with DART integration
//------------------------------------------------------------------------------

class BVH
{
public:
    BVH(const dart::dynamics::SkeletonPtr& skel, const std::map<std::string, std::string>& bvh_map);
    ~BVH();
    
    Eigen::VectorXd GetMotion(double t);
    Eigen::Matrix3d Get(const std::string& bvh_node);
    
    double GetMaxTime() { return mNumTotalFrames * mTimeStep; }
    double GetTimeStep() { return mTimeStep; }
    
    bool Parse(const std::string& file, std::string& errMsg, bool cyclic = true);
    
    const std::map<std::string, std::string>& GetBVHMap() { return mBVHMap; }
    const Eigen::Isometry3d& GetT0() { return T0; }
    const Eigen::Isometry3d& GetT1() { return T1; }
    bool IsCyclic() { return mCyclic; }
    int GetNumJoints() const { return static_cast<int>(mNodes.size()); }
    int GetNumFrames() const { return mNumTotalFrames; }

private:
    // Parsing helper functions (robust bvhview-style)
    void ParseWhitespace(Parser& par);
    bool ParseNewline(Parser& par);
    bool ParseString(Parser& par, const char* str);
    bool ParseFloat(Parser& par, double& out);
    bool ParseInt(Parser& par, int& out);
    bool ParseJointName(Parser& par, std::string& name);
    bool ParseOffset(Parser& par, Eigen::Vector3d& offset);
    bool ParseChannel(Parser& par, Channel& channel);
    bool ParseChannels(Parser& par, std::vector<Channel>& channels, int& channelCount);
    bool ParseJoint(Parser& par, BVHNode* parent, bool isRoot);
    bool ParseMotion(Parser& par);
    bool ParseHierarchy(Parser& par);
    
    // Data
    bool mCyclic;
    std::vector<Eigen::VectorXd> mMotions;
    std::map<std::string, BVHNode*> mMap;
    std::vector<BVHNode*> mNodes;  // For cleanup
    double mTimeStep;
    int mNumTotalChannels;
    int mNumTotalFrames;
    
    BVHNode* mRoot;
    
    dart::dynamics::SkeletonPtr mSkeleton;
    std::map<std::string, std::string> mBVHMap;
    
    Eigen::Isometry3d T0, T1;
};

}  // namespace MASS

#endif  // __MASS_BVH_H__