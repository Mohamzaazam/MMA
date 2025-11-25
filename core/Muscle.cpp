#include "Muscle.h"

using namespace MASS;
using namespace dart::dynamics;

std::vector<int> sort_indices(const std::vector<double>& val)
{
	std::vector<int> idx(val.size());
	std::iota(idx.begin(),idx.end(),0);

	std::sort(idx.begin(),idx.end(),[&val](int i1,int i2){return val[i1]<val[i2];});

	return idx;
}

Anchor::
Anchor(std::vector<BodyNode*> bns,std::vector<Eigen::Vector3d> lps,std::vector<double> ws)
	:bodynodes(bns),local_positions(lps),weights(ws),num_related_bodies(bns.size())
{

}

Eigen::Vector3d
Anchor::
GetPoint()
{
	Eigen::Vector3d p;
	p.setZero();
	for(int i = 0;i<num_related_bodies;i++)
		p += weights[i]*(bodynodes[i]->getTransform()*local_positions[i]);
	return p;
}

Muscle::
Muscle(std::string _name,double _f0,double _lm0,double _lt0,double _pen_angle,double lmax)
	:name(_name),f0(_f0),l_m0(_lm0),l_m(l_mt - l_t0),l_t0(_lt0),l_mt0(0.0),l_mt(1.0),
	 activation(0.0),f_toe(0.33),k_toe(3.0),k_lin(51.878788),e_toe(0.02),e_t0(0.033),
	 k_pe(4.0),e_mo(0.6),gamma(0.5),l_mt_max(lmax),
	 mCachesInitialized(false)  // OPTIMIZATION: Track cache initialization
{
}

void
Muscle::
InitializeCaches()
{
	if(mAnchors.empty()) return;
	
	const auto& skel = mAnchors[0]->bodynodes[0]->getSkeleton();
	int dof = skel->getNumDofs();
	int num_anchors = mAnchors.size();
	
	// OPTIMIZATION: Pre-allocate all matrices once
	mCachedJt.resize(dof, 3 * num_anchors);
	mCachedJt.setZero();
	
	mCachedA.resize(3 * num_anchors);
	mCachedA.setZero();
	
	mCachedP.resize(3 * num_anchors);
	mCachedP.setZero();
	
	mCachedJtA_reduced.resize(num_related_dofs);
	mCachedJtA_reduced.setZero();
	
	mCachesInitialized = true;
}

void
Muscle::
AddAnchor(const dart::dynamics::SkeletonPtr& skel,dart::dynamics::BodyNode* bn,const Eigen::Vector3d& glob_pos,int num_related_bodies)
{
	std::vector<double> distance;
	std::vector<Eigen::Vector3d> local_positions;
	distance.resize(skel->getNumBodyNodes(),0.0);
	local_positions.resize(skel->getNumBodyNodes());
	for(int i =0;i<skel->getNumBodyNodes();i++)
	{
		Eigen::Isometry3d T;
		T = skel->getBodyNode(i)->getTransform()*skel->getBodyNode(i)->getParentJoint()->getTransformFromChildBodyNode();
		local_positions[i] = skel->getBodyNode(i)->getTransform().inverse()*glob_pos;
		distance[i] = (glob_pos-T.translation()).norm();
	}
	std::vector<int> index_sort_by_distance = sort_indices(distance);
	
	std::vector<dart::dynamics::BodyNode*> lbs_body_nodes;
	std::vector<Eigen::Vector3d> lbs_local_positions;
	std::vector<double> lbs_weights;
		
	double total_weight = 0.0;
	if(distance[index_sort_by_distance[0]]<0.08)
	{
		lbs_weights.push_back(1.0/sqrt(distance[index_sort_by_distance[0]]));
		total_weight += lbs_weights[0];
		lbs_body_nodes.push_back(skel->getBodyNode(index_sort_by_distance[0]));
		lbs_local_positions.push_back(local_positions[index_sort_by_distance[0]]);

		if(lbs_body_nodes[0]->getParentBodyNode()!=nullptr)
		{
			auto bn_parent = lbs_body_nodes[0]->getParentBodyNode();
			lbs_weights.push_back(1.0/sqrt(distance[bn_parent->getIndexInSkeleton()]));
			total_weight += lbs_weights[1];
			lbs_body_nodes.push_back(bn_parent);
			lbs_local_positions.push_back(local_positions[bn_parent->getIndexInSkeleton()]);
		}	
	}
	else
	{
		total_weight = 1.0;
		lbs_weights.push_back(1.0);
		lbs_body_nodes.push_back(bn);
		lbs_local_positions.push_back(bn->getTransform().inverse()*glob_pos);
	}

	for(int i = 0;i < lbs_body_nodes.size();i++){
		lbs_weights[i] /= total_weight;
	}
	mAnchors.push_back(new Anchor(lbs_body_nodes,lbs_local_positions,lbs_weights));

	int n =mAnchors.size();
	if(n>1)
		l_mt0 += (mAnchors[n-1]->GetPoint()-mAnchors[n-2]->GetPoint()).norm();

	mCachedAnchorPositions.resize(n);
	Update();
	
	// Compute related DOFs
	Eigen::MatrixXd Jt_temp = skel->getNumDofs() > 0 ? 
		Eigen::MatrixXd::Zero(skel->getNumDofs(), 3*n) : Eigen::MatrixXd();
	
	for(int i = 0; i < n; i++)
		Jt_temp.block(0, i*3, skel->getNumDofs(), 3) = 
			skel->getLinearJacobian(mAnchors[i]->bodynodes[0],
				mAnchors[i]->bodynodes[0]->getTransform().inverse()*mCachedAnchorPositions[i]).transpose();
	
	double f_a_temp = Getf_A();
	Eigen::VectorXd JtA_temp = Eigen::VectorXd::Zero(skel->getNumDofs());
	
	for(int i = 0; i < n; i++)
	{
		Eigen::Vector3d dir = Eigen::Vector3d::Zero();
		if(i < n-1)
		{
			Eigen::Vector3d d = mCachedAnchorPositions[i+1]-mCachedAnchorPositions[i];
			d.normalize();
			dir += d;
		}
		if(i > 0)
		{
			Eigen::Vector3d d = mCachedAnchorPositions[i-1]-mCachedAnchorPositions[i];
			d.normalize();
			dir += d;
		}
		JtA_temp += Jt_temp.block(0, i*3, skel->getNumDofs(), 3) * (dir * f_a_temp);
	}
	
	num_related_dofs = 0;
	related_dof_indices.clear();
	for(int i = 0; i < JtA_temp.rows(); i++)
		if(std::abs(JtA_temp[i]) > 1E-3){
			num_related_dofs++;
			related_dof_indices.push_back(i);
		}
	
	// Initialize caches after determining related DOFs
	InitializeCaches();
}

void
Muscle::
AddAnchor(dart::dynamics::BodyNode* bn,const Eigen::Vector3d& glob_pos)
{
	std::vector<dart::dynamics::BodyNode*> lbs_body_nodes;
	std::vector<Eigen::Vector3d> lbs_local_positions;
	std::vector<double> lbs_weights;

	lbs_body_nodes.push_back(bn);
	lbs_local_positions.push_back(bn->getTransform().inverse()*glob_pos);
	lbs_weights.push_back(1.0);

	mAnchors.push_back(new Anchor(lbs_body_nodes,lbs_local_positions,lbs_weights));

	int n = mAnchors.size();
	if(n>1)
		l_mt0 += (mAnchors[n-1]->GetPoint()-mAnchors[n-2]->GetPoint()).norm();

	mCachedAnchorPositions.resize(n);
	Update();
	
	const auto& skel = bn->getSkeleton();
	
	// Compute related DOFs (same logic as above)
	Eigen::MatrixXd Jt_temp = Eigen::MatrixXd::Zero(skel->getNumDofs(), 3*n);
	for(int i = 0; i < n; i++)
		Jt_temp.block(0, i*3, skel->getNumDofs(), 3) = 
			skel->getLinearJacobian(mAnchors[i]->bodynodes[0],
				mAnchors[i]->bodynodes[0]->getTransform().inverse()*mCachedAnchorPositions[i]).transpose();
	
	double f_a_temp = Getf_A();
	Eigen::VectorXd JtA_temp = Eigen::VectorXd::Zero(skel->getNumDofs());
	
	for(int i = 0; i < n; i++)
	{
		Eigen::Vector3d dir = Eigen::Vector3d::Zero();
		if(i < n-1)
		{
			Eigen::Vector3d d = mCachedAnchorPositions[i+1]-mCachedAnchorPositions[i];
			d.normalize();
			dir += d;
		}
		if(i > 0)
		{
			Eigen::Vector3d d = mCachedAnchorPositions[i-1]-mCachedAnchorPositions[i];
			d.normalize();
			dir += d;
		}
		JtA_temp += Jt_temp.block(0, i*3, skel->getNumDofs(), 3) * (dir * f_a_temp);
	}
	
	num_related_dofs = 0;
	related_dof_indices.clear();
	for(int i = 0; i < JtA_temp.rows(); i++)
		if(std::abs(JtA_temp[i]) > 1E-3){
			num_related_dofs++;
			related_dof_indices.push_back(i);
		}
	
	InitializeCaches();
}

void
Muscle::
ApplyForceToBody()
{
	double f = GetForce();

	for(int i = 0; i < mAnchors.size()-1; i++)
	{
		Eigen::Vector3d dir = mCachedAnchorPositions[i+1]-mCachedAnchorPositions[i];
		dir.normalize();
		dir = f*dir;
		mAnchors[i]->bodynodes[0]->addExtForce(dir,mCachedAnchorPositions[i],false,false);
	}

	for(int i = 1; i < mAnchors.size(); i++)
	{
		Eigen::Vector3d dir = mCachedAnchorPositions[i-1]-mCachedAnchorPositions[i];
		dir.normalize();
		dir = f*dir;
		mAnchors[i]->bodynodes[0]->addExtForce(dir,mCachedAnchorPositions[i],false,false);
	}
}

void
Muscle::
Update()
{
	// OPTIMIZATION: Avoid function call overhead by inlining anchor position update
	for(int i = 0; i < mAnchors.size(); i++)
	{
		Eigen::Vector3d& p = mCachedAnchorPositions[i];
		p.setZero();
		Anchor* anchor = mAnchors[i];
		for(int j = 0; j < anchor->num_related_bodies; j++)
			p += anchor->weights[j] * (anchor->bodynodes[j]->getTransform() * anchor->local_positions[j]);
	}
	
	l_mt = Getl_mt();
	l_m = l_mt - l_t0;
}

double
Muscle::
GetForce()
{
	return Getf_A()*activation + Getf_p();
}

double
Muscle::
Getf_A()
{
	return f0*g_al(l_m/l_m0);
}

double
Muscle::
Getf_p()
{
	return f0*g_pl(l_m/l_m0);
}

double
Muscle::
Getl_mt()
{
	l_mt = 0.0;
	for(int i = 1; i < mAnchors.size(); i++)
		l_mt += (mCachedAnchorPositions[i]-mCachedAnchorPositions[i-1]).norm();

	return l_mt/l_mt0;
}

Eigen::VectorXd
Muscle::
GetRelatedJtA()
{
	// OPTIMIZATION: Use cached Jacobian transpose
	const Eigen::MatrixXd& Jt = GetJacobianTranspose();
	
	double f_a = Getf_A();
	
	// OPTIMIZATION: Compute JtA directly without intermediate full vector
	Eigen::VectorXd JtA = Eigen::VectorXd::Zero(Jt.rows());
	
	int num_anchors = mAnchors.size();
	for(int i = 0; i < num_anchors; i++)
	{
		Eigen::Vector3d dir = Eigen::Vector3d::Zero();
		if(i < num_anchors-1)
		{
			Eigen::Vector3d d = mCachedAnchorPositions[i+1]-mCachedAnchorPositions[i];
			d.normalize();
			dir += d;
		}
		if(i > 0)
		{
			Eigen::Vector3d d = mCachedAnchorPositions[i-1]-mCachedAnchorPositions[i];
			d.normalize();
			dir += d;
		}
		// OPTIMIZATION: Direct matrix-vector multiply on block
		JtA.noalias() += Jt.block(0, i*3, Jt.rows(), 3) * (dir * f_a);
	}
	
	// OPTIMIZATION: Use pre-allocated reduced vector
	for(int i = 0; i < num_related_dofs; i++){
		mCachedJtA_reduced[i] = JtA[related_dof_indices[i]];
	}

	return mCachedJtA_reduced;
}

const Eigen::MatrixXd&
Muscle::
GetJacobianTranspose()
{
	const auto& skel = mAnchors[0]->bodynodes[0]->getSkeleton();
	int dof = skel->getNumDofs();
	int num_anchors = mAnchors.size();
	
	// OPTIMIZATION: Reuse pre-allocated matrix
	if(!mCachesInitialized || mCachedJt.rows() != dof || mCachedJt.cols() != 3*num_anchors)
	{
		mCachedJt.resize(dof, 3*num_anchors);
	}
	
	mCachedJt.setZero();
	for(int i = 0; i < num_anchors; i++)
		mCachedJt.block(0, i*3, dof, 3) = skel->getLinearJacobian(
			mAnchors[i]->bodynodes[0],
			mAnchors[i]->bodynodes[0]->getTransform().inverse()*mCachedAnchorPositions[i]).transpose();
	
	return mCachedJt;	
}

std::pair<Eigen::VectorXd,Eigen::VectorXd>
Muscle::
GetForceJacobianAndPassive()
{
	double f_a = Getf_A();
	double f_p = Getf_p();
	
	int num_anchors = mAnchors.size();
	
	// OPTIMIZATION: Use pre-allocated vectors
	if(!mCachesInitialized || mCachedA.rows() != 3*num_anchors)
	{
		mCachedA.resize(3*num_anchors);
		mCachedP.resize(3*num_anchors);
	}
	
	mCachedA.setZero();
	mCachedP.setZero();

	for(int i = 0; i < num_anchors; i++)
	{
		Eigen::Vector3d dir = Eigen::Vector3d::Zero();
		
		if(i < num_anchors-1)
		{
			Eigen::Vector3d d = mCachedAnchorPositions[i+1]-mCachedAnchorPositions[i];
			d.normalize();
			dir += d;
		}
		if(i > 0)
		{
			Eigen::Vector3d d = mCachedAnchorPositions[i-1]-mCachedAnchorPositions[i];
			d.normalize();
			dir += d;
		}
		
		mCachedA.segment<3>(i*3) = dir * f_a;
		mCachedP.segment<3>(i*3) = dir * f_p;
	}
	
	return std::make_pair(mCachedA, mCachedP);
}

std::vector<dart::dynamics::Joint*>
Muscle::
GetRelatedJoints()
{
	auto skel = mAnchors[0]->bodynodes[0]->getSkeleton();
	std::map<dart::dynamics::Joint*,int> jns;
	std::vector<dart::dynamics::Joint*> jns_related;
	for(int i = 0; i < skel->getNumJoints(); i++)
		jns.insert(std::make_pair(skel->getJoint(i),0));

	Eigen::VectorXd dl_dtheta = Getdl_dtheta();
	
	for(int i = 0; i < dl_dtheta.rows(); i++)
		if(std::abs(dl_dtheta[i]) > 1E-6)
			jns[skel->getDof(i)->getJoint()]+=1;

	for(auto jn : jns)
		if(jn.second > 0)
			jns_related.push_back(jn.first);
	return jns_related;
}

std::vector<dart::dynamics::BodyNode*>
Muscle::
GetRelatedBodyNodes()
{
	std::vector<dart::dynamics::BodyNode*> bns_related;
	auto rjs = GetRelatedJoints();
	for(auto joint : rjs){
		bns_related.push_back(joint->getChildBodyNode());
	}

	return bns_related;
}

void
Muscle::
ComputeJacobians()
{
	const auto& skel = mAnchors[0]->bodynodes[0]->getSkeleton();
	int dof = skel->getNumDofs();
	mCachedJs.resize(mAnchors.size());
	for(int i = 0; i < mAnchors.size(); i++)
	{
		mCachedJs[i].resize(3, skel->getNumDofs());
		mCachedJs[i].setZero();

		for(int j = 0; j < mAnchors[i]->num_related_bodies; j++){
			mCachedJs[i] += mAnchors[i]->weights[j] * 
				skel->getLinearJacobian(mAnchors[i]->bodynodes[j], mAnchors[i]->local_positions[j]);
		}
	}
}

Eigen::VectorXd
Muscle::
Getdl_dtheta()
{
	ComputeJacobians();
	const auto& skel = mAnchors[0]->bodynodes[0]->getSkeleton();
	Eigen::VectorXd dl_dtheta(skel->getNumDofs());
	dl_dtheta.setZero();
	for(int i = 0; i < mAnchors.size()-1; i++)
	{
		Eigen::Vector3d pi = mCachedAnchorPositions[i+1] - mCachedAnchorPositions[i];
		Eigen::MatrixXd dpi_dtheta = mCachedJs[i+1] - mCachedJs[i];
		Eigen::VectorXd dli_d_theta = (dpi_dtheta.transpose()*pi)/(l_mt0*pi.norm());
		dl_dtheta += dli_d_theta;
	}

	for(int i = 0; i < dl_dtheta.rows(); i++)
		if(std::abs(dl_dtheta[i]) < 1E-6)
			dl_dtheta[i] = 0.0;

	return dl_dtheta;
}

double
Muscle::
g(double _l_m)
{
	double e_t = (l_mt -_l_m-l_t0)/l_t0;
	_l_m = _l_m/l_m0;
	double f = g_t(e_t) - (g_pl(_l_m)+activation*g_al(_l_m));
	return f;
}

double
Muscle::
g_t(double e_t)
{
	double f_t;
	if(e_t<=e_t0)
		f_t = f_toe/(exp(k_toe)-1)*(exp(k_toe*e_t/e_toe)-1);
	else
		f_t = k_lin*(e_t-e_toe)+f_toe;

	return f_t;
}

double
Muscle::
g_pl(double _l_m)
{
	double f_pl = (exp(k_pe*(_l_m-1.0)/e_mo)-1.0)/(exp(k_pe)-1.0);
	if(_l_m<1.0)
		return 0.0;
	else
		return f_pl;
}

double
Muscle::
g_al(double _l_m)
{
	return exp(-(_l_m-1.0)*(_l_m-1.0)/gamma);
}