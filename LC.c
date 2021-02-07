#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <string.h>
#include <stdbool.h>
#include <vld.h>

//2021_2_7
//110. 平衡二叉树
//* Definition for a binary tree node.
//* struct TreeNode {
//	*int val;
//	*struct TreeNode *left;
//	*struct TreeNode *right;
//	*
//};
//*/
//
//int _maxdepth(struct TreeNode* root)
//{
//	if (root == NULL)
//		return 0;//return; [1]这种情况不符
//
//	int left = _maxdepth(root->left);
//	int right = _maxdepth(root->right);
//
//	return (left > right ? left : right) + 1;
//
//}
//
//bool isBalanced(struct TreeNode* root){
//
//	if (root == NULL)
//		return true;
//
//	int lhigh = _maxdepth(root->left);
//	int rhigh = _maxdepth(root->right);
//
//	return abs(rhigh - lhigh) <2 && isBalanced(root->left) && isBalanced(root->right);//每个节点都要满足差值<=1
//}

//572. 另一个树的子树
///**
//* Definition for a binary tree node.
//* struct TreeNode {
//*     int val;
//*     struct TreeNode *left;
//*     struct TreeNode *right;
//* };
//*/
//
//bool _IsSame(struct TreeNode* s, struct TreeNode* t){
//
//	if (s == NULL && t == NULL)
//		return true;
//	if (s == NULL || t == NULL)
//		return false;
//
//	return s->val == t->val && _IsSame(s->left, t->left) && _IsSame(s->right, t->right);
//}
//
//bool isSubtree(struct TreeNode* s, struct TreeNode* t){
//
//	if (t == NULL)
//		return true;
//
//	if (s == NULL)
//		return false;
//
//	if (_IsSame(s, t))
//		return true;
//
//	//return isSubtree(s->left, t->right) || isSubtree(s->left, t->right);//错误
//	return isSubtree(s->left, t) || isSubtree(s->right, t);
//}

//101. 对称二叉树
///**
//* Definition for a binary tree node.
//* struct TreeNode {
//*     int val;
//*     struct TreeNode *left;
//*     struct TreeNode *right;
//* };
//*/
//bool _isLRSame(struct TreeNode* t1, struct TreeNode* t2)
//{
//	if (t1 == NULL && t2 == NULL)
//		return true;
//	//if(t1==NULL || t2==NULL);    
//	if (t1 == NULL || t2 == NULL)
//		return false;
//
//	return t1->val == t2->val && _isLRSame(t1->left, t2->right) && _isLRSame(t1->right, t2->left);
//}
//
//bool isSymmetric(struct TreeNode* root){
//
//	if (root == NULL)
//		return true;
//
//	return _isLRSame(root->left, root->right);
//}

//100. 相同的树
///**
//* Definition for a binary tree node.
//* struct TreeNode {
//*     int val;
//*     struct TreeNode *left;
//*     struct TreeNode *right;
//* };
//*/
//
//
//bool isSameTree(struct TreeNode* p, struct TreeNode* q){
//
//	if (p == NULL && q == NULL)
//		return true;
//	if (p == NULL || q == NULL)
//		return false;
//
//	return p->val == q->val && isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
//}

//226. 翻转二叉树
///**
//* Definition for a binary tree node.
//* struct TreeNode {
//*     int val;
//*     struct TreeNode *left;
//*     struct TreeNode *right;
//* };
//*/
//
//
//struct TreeNode* invertTree(struct TreeNode* root){
//
//	if (root == NULL)
//		return NULL;
//
//	struct TreeNode* _left = invertTree(root->left);
//	struct TreeNode* _right = invertTree(root->right);
//
//	root->left = _right;
//	root->right = _left;
//
//	return root;
//}

//104. 二叉树的最大深度
///**
//* Definition for a binary tree node.
//* struct TreeNode {
//*     int val;
//*     struct TreeNode *left;
//*     struct TreeNode *right;
//* };
//*/
//
//
//int maxDepth(struct TreeNode* root){
//
//	if (root == NULL)
//		return 0;
//
//	int left = maxDepth(root->left);
//	int right = maxDepth(root->right);
//
//	//return left > right ? (left : right) +1
//	return (left > right ? left : right) + 1;
//}

//965. 单值二叉树
///**
//* Definition for a binary tree node.
//* struct TreeNode {
//*     int val;
//*     struct TreeNode *left;
//*     struct TreeNode *right;
//* };
//*/
//
//
//bool isUnivalTree(struct TreeNode* root){
//
//	if (root == NULL)
//		return true;
//	if (root->left && root->left->val != root->val)
//		return false;
//	if (root->right && root->right->val != root->val)
//		return false;
//	return isUnivalTree(root->left) && isUnivalTree(root->right);
//}

//144. 二叉树的前序遍历
///**
//* Definition for a binary tree node.
//* struct TreeNode {
//*     int val;
//*     struct TreeNode *left;
//*     struct TreeNode *right;
//* };
//*/
//
//
///**
//* Note: The returned array must be malloced, assume caller calls free().
//*/
//int size(struct TreeNode* root)
//{
//	if (root == NULL)
//		return 0;
//
//	int left = size(root->left);
//	int right = size(root->right);
//
//	return left + right + 1;
//}
//
//int* _preorderTraversal(struct TreeNode* root, int* p, int *index)
//{
//	if (root != NULL){
//
//		p[*index] = root->val;
//		(*index)++;
//		_preorderTraversal(root->left, p, index);
//		_preorderTraversal(root->right, p, index);
//	}
//	return p;
//}
//
//int* preorderTraversal(struct TreeNode* root, int* returnSize){
//
//	int n = size(root);
//	int* p = (int*)malloc(sizeof(int)*n);
//
//	int index = 0;
//	//_preorderTraversal(root, p, index);
//	_preorderTraversal(root, p, &index);
//
//	//
//	*returnSize = n;
//	return p;
//}

//94. 二叉树的中序遍历
///**
//* Definition for a binary tree node.
//* struct TreeNode {
//*     int val;
//*     struct TreeNode *left;
//*     struct TreeNode *right;
//* };
//*/
//
//
///**
//* Note: The returned array must be malloced, assume caller calls free().
//*/
//int size(struct TreeNode* root)
//{
//	if (root == NULL)
//		return 0;
//
//	int left = size(root->left);
//	int right = size(root->right);
//
//	return left + right + 1;
//}
//
//int* _inorderTraversal(struct TreeNode* root, int* p, int* index)
//{
//	if (root != NULL){
//
//		_inorderTraversal(root->left, p, index);
//		p[*index] = root->val;
//		(*index)++;
//		_inorderTraversal(root->right, p, index);
//	}
//	return p;
//}
//
//int* inorderTraversal(struct TreeNode* root, int* returnSize){
//
//	int n = size(root);
//	int *p = (int*)malloc(sizeof(int)*n);
//	*returnSize = n;
//
//	int index = 0;
//	_inorderTraversal(root, p, &index);
//
//	return p;
//}

//145. 二叉树的后序遍历
///**
//* Definition for a binary tree node.
//* struct TreeNode {
//*     int val;
//*     struct TreeNode *left;
//*     struct TreeNode *right;
//* };
//*/
//
//
///**
//* Note: The returned array must be malloced, assume caller calls free().
//*/
//
//int size(struct TreeNode* root)
//{
//	if (root == NULL)
//		return 0;
//
//	int left = size(root->left);
//	int right = size(root->right);
//
//	return left + right + 1;
//}
//
//int* _postorderTraversal(struct TreeNode* root, int *p, int* index)
//{
//	if (root != NULL){
//
//		_postorderTraversal(root->left, p, index);
//		_postorderTraversal(root->right, p, index);
//		p[(*index)++] = root->val;
//
//	}
//	return p;
//}
//
//int* postorderTraversal(struct TreeNode* root, int* returnSize){
//
//	int n = size(root);
//	int *p = (int*)malloc(sizeof(int)*n);
//	*returnSize = n;
//
//	int index = 0;
//	_postorderTraversal(root, p, &index);
//
//	return p;
//}
int main()
{
	EXIT_SUCCESS;
}