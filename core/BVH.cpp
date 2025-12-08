#include "BVH.h"
#include <iostream>
#include <cstdarg>
#include <algorithm>

using namespace dart::dynamics;

namespace MASS
{

//------------------------------------------------------------------------------
// Rotation Helper Functions
//------------------------------------------------------------------------------

Eigen::Matrix3d R_x(double x)
{
    double cosa = cos(x * M_PI / 180.0);
    double sina = sin(x * M_PI / 180.0);
    Eigen::Matrix3d R;
    R << 1, 0, 0,
         0, cosa, -sina,
         0, sina, cosa;
    return R;
}

Eigen::Matrix3d R_y(double y)
{
    double cosa = cos(y * M_PI / 180.0);
    double sina = sin(y * M_PI / 180.0);
    Eigen::Matrix3d R;
    R << cosa, 0, sina,
         0, 1, 0,
         -sina, 0, cosa;
    return R;
}

Eigen::Matrix3d R_z(double z)
{
    double cosa = cos(z * M_PI / 180.0);
    double sina = sin(z * M_PI / 180.0);
    Eigen::Matrix3d R;
    R << cosa, -sina, 0,
         sina, cosa, 0,
         0, 0, 1;
    return R;
}

//------------------------------------------------------------------------------
// Parser Implementation
//------------------------------------------------------------------------------

void Parser::Init(const std::string& fname, const std::string& content)
{
    filename = fname;
    offset = 0;
    data = content;
    row = 1;
    col = 1;
    err[0] = '\0';
}

char Parser::Peek() const
{
    return offset < static_cast<int>(data.size()) ? data[offset] : '\0';
}

char Parser::PeekForward(int steps) const
{
    int idx = offset + steps;
    return idx < static_cast<int>(data.size()) ? data[idx] : '\0';
}

bool Parser::Match(char c) const
{
    return Peek() == c;
}

bool Parser::OneOf(const char* chars) const
{
    char c = Peek();
    return c != '\0' && strchr(chars, c) != nullptr;
}

bool Parser::StartsWithCaseless(const char* prefix) const
{
    const char* start = data.c_str() + offset;
    while (*prefix)
    {
        if (tolower(*prefix) != tolower(*start)) return false;
        prefix++;
        start++;
    }
    return true;
}

void Parser::Inc()
{
    if (offset < static_cast<int>(data.size()))
    {
        if (data[offset] == '\n')
        {
            row++;
            col = 1;
        }
        else
        {
            col++;
        }
        offset++;
    }
}

void Parser::Advance(int num)
{
    for (int i = 0; i < num; i++) Inc();
}

const char* Parser::CharName(char c)
{
    static char buf[2];
    switch (c)
    {
        case '\0': return "end of file";
        case '\r': return "carriage return";
        case '\n': return "newline";
        case '\t': return "tab";
        default:
            buf[0] = c;
            buf[1] = '\0';
            return buf;
    }
}

void Parser::SetError(const char* fmt, ...)
{
    char buffer[256];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    
    snprintf(err, ERR_MAX, "%s:%d:%d: error: %s", 
             filename.c_str(), row, col, buffer);
}

//------------------------------------------------------------------------------
// BVHNode Implementation
//------------------------------------------------------------------------------

BVHNode::BVHNode(const std::string& name, BVHNode* parent)
    : mParent(parent), mName(name), mOffset(Eigen::Vector3d::Zero()),
      mEndSite(false), mChannelOffset(0), mNumChannels(0)
{
    mR.setIdentity();
}

void BVHNode::SetChannel(int c_offset, const std::vector<Channel>& channels)
{
    mChannelOffset = c_offset;
    mNumChannels = static_cast<int>(channels.size());
    mChannels = channels;
}

void BVHNode::Set(const Eigen::VectorXd& m_t)
{
    mR.setIdentity();
    
    // Apply rotations in the order specified in the BVH file
    for (int i = 0; i < mNumChannels; i++)
    {
        double val = m_t[mChannelOffset + i];
        switch (mChannels[i])
        {
            case CHANNEL_X_POSITION: break;  // Position handled separately
            case CHANNEL_Y_POSITION: break;
            case CHANNEL_Z_POSITION: break;
            case CHANNEL_X_ROTATION: mR = mR * R_x(val); break;
            case CHANNEL_Y_ROTATION: mR = mR * R_y(val); break;
            case CHANNEL_Z_ROTATION: mR = mR * R_z(val); break;
            default: break;
        }
    }
}

void BVHNode::Set(const Eigen::Matrix3d& R_t)
{
    mR = R_t;
}

Eigen::Matrix3d BVHNode::Get() const
{
    return mR;
}

void BVHNode::AddChild(BVHNode* child)
{
    mChildren.push_back(child);
}

BVHNode* BVHNode::GetNode(const std::string& name)
{
    if (mName == name)
        return this;
    
    for (auto& c : mChildren)
    {
        BVHNode* bn = c->GetNode(name);
        if (bn != nullptr)
            return bn;
    }
    
    return nullptr;
}

//------------------------------------------------------------------------------
// BVH Implementation - Parsing Functions
//------------------------------------------------------------------------------

BVH::BVH(const SkeletonPtr& skel, const std::map<std::string, std::string>& bvh_map)
    : mCyclic(true), mTimeStep(0.0), mNumTotalChannels(0), mNumTotalFrames(0),
      mRoot(nullptr), mSkeleton(skel), mBVHMap(bvh_map)
{
    T0.setIdentity();
    T1.setIdentity();
}

BVH::~BVH()
{
    for (auto* node : mNodes)
    {
        delete node;
    }
    mNodes.clear();
    mMap.clear();
}

void BVH::ParseWhitespace(Parser& par)
{
    while (par.OneOf(" \r\t\v")) par.Inc();
}

bool BVH::ParseNewline(Parser& par)
{
    ParseWhitespace(par);
    
    if (par.Match('\n'))
    {
        par.Inc();
        ParseWhitespace(par);
        return true;
    }
    else
    {
        par.SetError("expected newline at '%s'", par.CharName(par.Peek()));
        return false;
    }
}

bool BVH::ParseString(Parser& par, const char* str)
{
    ParseWhitespace(par);
    
    if (par.StartsWithCaseless(str))
    {
        par.Advance(strlen(str));
        ParseWhitespace(par);
        return true;
    }
    else
    {
        par.SetError("expected '%s' at '%s'", str, par.CharName(par.Peek()));
        return false;
    }
}

bool BVH::ParseFloat(Parser& par, double& out)
{
    ParseWhitespace(par);
    
    char* end;
    errno = 0;
    out = strtod(par.data.c_str() + par.offset, &end);
    
    if (errno == 0 && end != par.data.c_str() + par.offset)
    {
        par.Advance(end - (par.data.c_str() + par.offset));
        return true;
    }
    else
    {
        par.SetError("expected float at '%s'", par.CharName(par.Peek()));
        return false;
    }
}

bool BVH::ParseInt(Parser& par, int& out)
{
    ParseWhitespace(par);
    
    char* end;
    errno = 0;
    out = static_cast<int>(strtol(par.data.c_str() + par.offset, &end, 10));
    
    if (errno == 0 && end != par.data.c_str() + par.offset)
    {
        par.Advance(end - (par.data.c_str() + par.offset));
        return true;
    }
    else
    {
        par.SetError("expected integer at '%s'", par.CharName(par.Peek()));
        return false;
    }
}

bool BVH::ParseJointName(Parser& par, std::string& name)
{
    ParseWhitespace(par);
    
    name.clear();
    const char* validChars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_:-.";
    
    while (par.OneOf(validChars))
    {
        name += par.Peek();
        par.Inc();
    }
    
    if (!name.empty())
    {
        ParseWhitespace(par);
        return true;
    }
    else
    {
        par.SetError("expected joint name at '%s'", par.CharName(par.Peek()));
        return false;
    }
}

bool BVH::ParseOffset(Parser& par, Eigen::Vector3d& offset)
{
    if (!ParseString(par, "OFFSET")) return false;
    if (!ParseFloat(par, offset.x())) return false;
    if (!ParseFloat(par, offset.y())) return false;
    if (!ParseFloat(par, offset.z())) return false;
    if (!ParseNewline(par)) return false;
    return true;
}

bool BVH::ParseChannel(Parser& par, Channel& channel)
{
    ParseWhitespace(par);
    
    if (par.Peek() == '\0')
    {
        par.SetError("expected channel at end of file");
        return false;
    }
    
    char first = par.Peek();
    char second = par.PeekForward(1);
    
    // Position channels
    if (first == 'X' && second == 'p')
    {
        if (!ParseString(par, "Xposition")) return false;
        channel = CHANNEL_X_POSITION;
        return true;
    }
    if (first == 'Y' && second == 'p')
    {
        if (!ParseString(par, "Yposition")) return false;
        channel = CHANNEL_Y_POSITION;
        return true;
    }
    if (first == 'Z' && second == 'p')
    {
        if (!ParseString(par, "Zposition")) return false;
        channel = CHANNEL_Z_POSITION;
        return true;
    }
    
    // Rotation channels
    if (first == 'X' && second == 'r')
    {
        if (!ParseString(par, "Xrotation")) return false;
        channel = CHANNEL_X_ROTATION;
        return true;
    }
    if (first == 'Y' && second == 'r')
    {
        if (!ParseString(par, "Yrotation")) return false;
        channel = CHANNEL_Y_ROTATION;
        return true;
    }
    if (first == 'Z' && second == 'r')
    {
        if (!ParseString(par, "Zrotation")) return false;
        channel = CHANNEL_Z_ROTATION;
        return true;
    }
    
    par.SetError("expected channel type (Xposition, Yrotation, etc.)");
    return false;
}

bool BVH::ParseChannels(Parser& par, std::vector<Channel>& channels, int& channelCount)
{
    if (!ParseString(par, "CHANNELS")) return false;
    if (!ParseInt(par, channelCount)) return false;
    
    channels.resize(channelCount);
    for (int i = 0; i < channelCount; i++)
    {
        if (!ParseChannel(par, channels[i])) return false;
    }
    
    if (!ParseNewline(par)) return false;
    return true;
}

bool BVH::ParseJoint(Parser& par, BVHNode* parent, bool isRoot)
{
    std::string name;
    
    // Parse joint/root keyword and name
    if (isRoot)
    {
        if (!ParseString(par, "ROOT")) return false;
    }
    else
    {
        // Check for End Site or JOINT
        ParseWhitespace(par);
        if (par.StartsWithCaseless("End"))
        {
            if (!ParseString(par, "End")) return false;
            if (!ParseString(par, "Site")) return false;
            name = "EndSite_" + (parent ? parent->GetName() : "unknown");
            
            BVHNode* node = new BVHNode(name, parent);
            node->SetEndSite(true);
            mNodes.push_back(node);
            mMap[name] = node;
            if (parent) parent->AddChild(node);
            
            if (!ParseNewline(par)) return false;
            if (!ParseString(par, "{")) return false;
            if (!ParseNewline(par)) return false;
            
            Eigen::Vector3d offset;
            if (!ParseOffset(par, offset)) return false;
            node->SetOffset(offset);
            
            if (!ParseString(par, "}")) return false;
            if (!ParseNewline(par)) return false;
            
            return true;
        }
        
        if (!ParseString(par, "JOINT")) return false;
    }
    
    if (!ParseJointName(par, name)) return false;
    if (!ParseNewline(par)) return false;
    if (!ParseString(par, "{")) return false;
    if (!ParseNewline(par)) return false;
    
    // Create node
    BVHNode* node = new BVHNode(name, parent);
    mNodes.push_back(node);
    mMap[name] = node;
    if (parent) parent->AddChild(node);
    if (isRoot) mRoot = node;
    
    // Parse offset
    Eigen::Vector3d offset;
    if (!ParseOffset(par, offset)) return false;
    node->SetOffset(offset);
    
    // Parse channels
    std::vector<Channel> channels;
    int channelCount;
    if (!ParseChannels(par, channels, channelCount)) return false;
    node->SetChannel(mNumTotalChannels, channels);
    mNumTotalChannels += channelCount;
    
    // Parse child joints
    ParseWhitespace(par);
    while (par.StartsWithCaseless("JOINT") || par.StartsWithCaseless("End"))
    {
        if (!ParseJoint(par, node, false)) return false;
        ParseWhitespace(par);
    }
    
    if (!ParseString(par, "}")) return false;
    if (!ParseNewline(par)) return false;
    
    return true;
}

bool BVH::ParseHierarchy(Parser& par)
{
    if (!ParseString(par, "HIERARCHY")) return false;
    if (!ParseNewline(par)) return false;
    
    mNumTotalChannels = 0;
    if (!ParseJoint(par, nullptr, true)) return false;
    
    return true;
}

bool BVH::ParseMotion(Parser& par)
{
    if (!ParseString(par, "MOTION")) return false;
    if (!ParseNewline(par)) return false;
    
    // Parse frame count
    if (!ParseString(par, "Frames:")) return false;
    if (!ParseInt(par, mNumTotalFrames)) return false;
    if (!ParseNewline(par)) return false;
    
    // Parse frame time
    if (!ParseString(par, "Frame")) return false;
    if (!ParseString(par, "Time:")) return false;
    if (!ParseFloat(par, mTimeStep)) return false;
    if (!ParseNewline(par)) return false;
    
    if (mTimeStep == 0.0) mTimeStep = 1.0 / 60.0;
    
    // Parse motion data
    mMotions.resize(mNumTotalFrames);
    for (int i = 0; i < mNumTotalFrames; i++)
    {
        mMotions[i] = Eigen::VectorXd::Zero(mNumTotalChannels);
        for (int j = 0; j < mNumTotalChannels; j++)
        {
            if (!ParseFloat(par, mMotions[i][j])) return false;
        }
        
        // Handle optional newline at end of file
        ParseWhitespace(par);
        if (par.Match('\n'))
        {
            par.Inc();
            ParseWhitespace(par);
        }
    }
    
    return true;
}

bool BVH::Parse(const std::string& file, std::string& errMsg, bool cyclic)
{
    mCyclic = cyclic;
    
    // Read file contents
    std::ifstream ifs(file, std::ios::binary);
    if (!ifs)
    {
        errMsg = "Could not open file: " + file;
        return false;
    }
    
    std::string content((std::istreambuf_iterator<char>(ifs)),
                        std::istreambuf_iterator<char>());
    ifs.close();
    
    // Append newline if not present
    if (!content.empty() && content.back() != '\n')
    {
        content += '\n';
    }
    
    // Clear any existing data
    for (auto* node : mNodes) delete node;
    mNodes.clear();
    mMap.clear();
    mMotions.clear();
    mRoot = nullptr;
    mNumTotalChannels = 0;
    mNumTotalFrames = 0;
    
    // Parse
    Parser par;
    par.Init(file, content);
    
    if (!ParseHierarchy(par))
    {
        errMsg = par.err;
        return false;
    }
    
    if (!ParseMotion(par))
    {
        errMsg = par.err;
        return false;
    }
    
    // Set T0 and T1 for cyclic motion
    if (!mBVHMap.empty() && mSkeleton)
    {
        BodyNode* root = mSkeleton->getRootBodyNode();
        if (root && mBVHMap.count(root->getName()))
        {
            std::string root_bvh_name = mBVHMap.at(root->getName());
            if (mMap.count(root_bvh_name))
            {
                Eigen::VectorXd m = mMotions[0];
                mMap[root_bvh_name]->Set(m);
                T0.linear() = Get(root_bvh_name);
                T0.translation() = 0.01 * m.segment<3>(0);
                
                m = mMotions[mNumTotalFrames - 1];
                mMap[root_bvh_name]->Set(m);
                T1.linear() = Get(root_bvh_name);
                T1.translation() = 0.01 * m.segment<3>(0);
            }
        }
    }
    
    std::cout << "INFO: Parsed '" << file << "' successfully (" 
              << mNodes.size() << " joints, " 
              << mNumTotalFrames << " frames)" << std::endl;
    
    errMsg.clear();
    return true;
}

//------------------------------------------------------------------------------
// BVH Implementation - DART Integration
//------------------------------------------------------------------------------

Eigen::Matrix3d BVH::Get(const std::string& bvh_node)
{
    auto it = mMap.find(bvh_node);
    if (it != mMap.end())
    {
        return it->second->Get();
    }
    return Eigen::Matrix3d::Identity();
}

Eigen::VectorXd BVH::GetMotion(double t)
{
    int k = static_cast<int>(std::floor(t / mTimeStep));
    if (mCyclic)
        k %= mNumTotalFrames;
    k = std::max(0, std::min(k, mNumTotalFrames - 1));
    
    Eigen::VectorXd m_t = mMotions[k];
    
    // Set rotation for each node
    for (auto& bn : mMap)
    {
        bn.second->Set(m_t);
    }
    
    // Map to DART skeleton
    int dof = mSkeleton->getNumDofs();
    Eigen::VectorXd p = Eigen::VectorXd::Zero(dof);
    
    for (const auto& ss : mBVHMap)
    {
        BodyNode* bn = mSkeleton->getBodyNode(ss.first);
        if (!bn) continue;
        
        Eigen::Matrix3d R = Get(ss.second);
        Joint* jn = bn->getParentJoint();
        int idx = jn->getIndexInSkeleton(0);
        
        if (jn->getType() == "FreeJoint")
        {
            Eigen::Isometry3d T;
            T.translation() = 0.01 * m_t.segment<3>(0);  // cm to m
            T.linear() = R;
            p.segment<6>(idx) = FreeJoint::convertToPositions(T);
        }
        else if (jn->getType() == "BallJoint")
        {
            p.segment<3>(idx) = BallJoint::convertToPositions(R);
        }
        else if (jn->getType() == "RevoluteJoint")
        {
            Eigen::Vector3d u = dynamic_cast<RevoluteJoint*>(jn)->getAxis();
            Eigen::Vector3d aa = BallJoint::convertToPositions(R);
            double val;
            if ((u - Eigen::Vector3d::UnitX()).norm() < 1E-4)
                val = aa[0];
            else if ((u - Eigen::Vector3d::UnitY()).norm() < 1E-4)
                val = aa[1];
            else
                val = aa[2];
            
            // Normalize to [-PI, PI]
            if (val > M_PI)
                val -= 2 * M_PI;
            else if (val < -M_PI)
                val += 2 * M_PI;
            
            p[idx] = val;
        }
    }
    
    return p;
}

}  // namespace MASS