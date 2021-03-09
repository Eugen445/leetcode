//#define _CRT_SECURE_
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <string.h>
#include <stdbool.h>
#include <vld.h>

//2021_2_7
//110. 平衡二叉树
* Definition for a binary tree node.
* struct TreeNode {
	*int val;
	*struct TreeNode *left;
	*struct TreeNode *right;
	*
};
*/

int _maxdepth(struct TreeNode* root)
{ 
	if (root == NULL)
		return 0;//return; [1]这种情况不符

	int left = _maxdepth(root->left);
	int right = _maxdepth(root->right);

	return (left > right ? left : right) + 1;

}

bool isBalanced(struct TreeNode* root){

	if (root == NULL)
		return true;

	int lhigh = _maxdepth(root->left);
	int rhigh = _maxdepth(root->right);

	return abs(rhigh - lhigh) <2 && isBalanced(root->left) && isBalanced(root->right);//每个节点都要满足差值<=1
}

//572. 另一个树的子树
/**
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     struct TreeNode *left;
*     struct TreeNode *right;
* };
*/

bool _IsSame(struct TreeNode* s, struct TreeNode* t){

	if (s == NULL && t == NULL)
		return true;
	if (s == NULL || t == NULL)
		return false;

	return s->val == t->val && _IsSame(s->left, t->left) && _IsSame(s->right, t->right);
}

bool isSubtree(struct TreeNode* s, struct TreeNode* t){

	if (t == NULL)
		return true;

	if (s == NULL)
		return false;

	if (_IsSame(s, t))
		return true;

	//return isSubtree(s->left, t->right) || isSubtree(s->left, t->right);//错误
	return isSubtree(s->left, t) || isSubtree(s->right, t);
}

//101. 对称二叉树
/**
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     struct TreeNode *left;
*     struct TreeNode *right;
* };
*/
bool _isLRSame(struct TreeNode* t1, struct TreeNode* t2)
{
	if (t1 == NULL && t2 == NULL)
		return true;
	//if(t1==NULL || t2==NULL);    
	if (t1 == NULL || t2 == NULL)
		return false;

	return t1->val == t2->val && _isLRSame(t1->left, t2->right) && _isLRSame(t1->right, t2->left);
}

bool isSymmetric(struct TreeNode* root){

	if (root == NULL)
		return true;

	return _isLRSame(root->left, root->right);
}

//100. 相同的树
/**
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     struct TreeNode *left;
*     struct TreeNode *right;
* };
*/


bool isSameTree(struct TreeNode* p, struct TreeNode* q){

	if (p == NULL && q == NULL)
		return true;
	if (p == NULL || q == NULL)
		return false;

	return p->val == q->val && isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
}

//226. 翻转二叉树
/**
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     struct TreeNode *left;
*     struct TreeNode *right;
* };
*/


struct TreeNode* invertTree(struct TreeNode* root){

	if (root == NULL)
		return NULL;

	struct TreeNode* _left = invertTree(root->left);
	struct TreeNode* _right = invertTree(root->right);

	root->left = _right;
	root->right = _left;

	return root;
}

//104. 二叉树的最大深度
/**
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     struct TreeNode *left;
*     struct TreeNode *right;
* };
*/


int maxDepth(struct TreeNode* root){

	if (root == NULL)
		return 0;

	int left = maxDepth(root->left);
	int right = maxDepth(root->right);

	//return left > right ? (left : right) +1
	return (left > right ? left : right) + 1;
}

//965. 单值二叉树
/**
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     struct TreeNode *left;
*     struct TreeNode *right;
* };
*/


bool isUnivalTree(struct TreeNode* root){

	if (root == NULL)
		return true;
	if (root->left && root->left->val != root->val)
		return false;
	if (root->right && root->right->val != root->val)
		return false;
	return isUnivalTree(root->left) && isUnivalTree(root->right);
}

//144. 二叉树的前序遍历
/**
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     struct TreeNode *left;
*     struct TreeNode *right;
* };
*/


/**
* Note: The returned array must be malloced, assume caller calls free().
*/
int size(struct TreeNode* root)
{
	if (root == NULL)
		return 0;

	int left = size(root->left);
	int right = size(root->right);

	return left + right + 1;
}

int* _preorderTraversal(struct TreeNode* root, int* p, int *index)
{
	if (root != NULL){

		p[*index] = root->val;
		(*index)++;
		_preorderTraversal(root->left, p, index);
		_preorderTraversal(root->right, p, index);
	}
	return p;
}

int* preorderTraversal(struct TreeNode* root, int* returnSize){

	int n = size(root);
	int* p = (int*)malloc(sizeof(int)*n);

	int index = 0;
	//_preorderTraversal(root, p, index);
	_preorderTraversal(root, p, &index);

	//
	*returnSize = n;
	return p;
}

//94. 二叉树的中序遍历
/**
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     struct TreeNode *left;
*     struct TreeNode *right;
* };
*/


/**
* Note: The returned array must be malloced, assume caller calls free().
*/
int size(struct TreeNode* root)
{
	if (root == NULL)
		return 0;

	int left = size(root->left);
	int right = size(root->right);

	return left + right + 1;
}

int* _inorderTraversal(struct TreeNode* root, int* p, int* index)
{
	if (root != NULL){

		_inorderTraversal(root->left, p, index);
		p[*index] = root->val;
		(*index)++;
		_inorderTraversal(root->right, p, index);
	}
	return p;
}

int* inorderTraversal(struct TreeNode* root, int* returnSize){

	int n = size(root);
	int *p = (int*)malloc(sizeof(int)*n);
	*returnSize = n;

	int index = 0;
	_inorderTraversal(root, p, &index);

	return p;
}

//145. 二叉树的后序遍历
/**
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     struct TreeNode *left;
*     struct TreeNode *right;
* };
*/


/**
* Note: The returned array must be malloced, assume caller calls free().
*/

int size(struct TreeNode* root)
{
	if (root == NULL)
		return 0;

	int left = size(root->left);
	int right = size(root->right);

	return left + right + 1;
}

int* _postorderTraversal(struct TreeNode* root, int *p, int* index)
{
	if (root != NULL){

		_postorderTraversal(root->left, p, index);
		_postorderTraversal(root->right, p, index);
		p[(*index)++] = root->val;

	}
	return p;
}

int* postorderTraversal(struct TreeNode* root, int* returnSize){

	int n = size(root);
	int *p = (int*)malloc(sizeof(int)*n);
	*returnSize = n;

	int index = 0;
	_postorderTraversal(root, p, &index);

	return p;
}

//2021_2_10
//225. 用队列实现栈
#define QueueElemType int

//链队列
typedef struct ChainQueueNode
{
	QueueElemType data;
	struct ChainQueueNode* next;
}CQN;

typedef struct ChainQueue
{
	CQN* head;
	CQN* tail;
}ChainQueue;

void ChainQueueInit(ChainQueue *cq);
void ChainQueueDestroy(ChainQueue *cq);
void ChainQueueEn(ChainQueue *cq, QueueElemType x);
void ChainQueueDe(ChainQueue *cq);
void ChainQueueShow(ChainQueue *cq);
QueueElemType ChainQueueFront(ChainQueue *cq);
QueueElemType ChainQueueBack(ChainQueue *cq);

bool IsEmpty(ChainQueue *cq)
{
	return cq->head == NULL;
}

QueueElemType ChainQueueBack(ChainQueue *cq);

void ChainQueueInit(ChainQueue *cq)
{
	assert(cq);
	cq->head = cq->tail = NULL;
}

void ChainQueueDestroy(ChainQueue *cq)
{
	assert(cq);
	while (cq->head){
		CQN *p = cq->head;
		cq->head = p->next;
		free(p);
	}
	//
	cq->head = cq->tail = NULL;
}

void ChainQueueEn(ChainQueue *cq, QueueElemType x)
{
	assert(cq);
	CQN *s = (CQN*)malloc(sizeof(CQN));
	//
	assert(s != NULL);
	s->data = x;
	s->next = NULL;

	if (cq->head == NULL)//无元素
		cq->head = cq->tail = s;
	else{
		cq->tail->next = s;
		cq->tail = s;
	}
}

void ChainQueueDe(ChainQueue *cq)
{
	assert(cq);
	//if (cq->head == NULL){
	//	printf("队列已空\n");
	//	return;
	//}
	if (IsEmpty(cq)){
		printf("队列已空\n");
		return;
	}
	CQN *p = cq->head;
	cq->head = p->next;
	free(p);
	//
	if (cq->head == NULL)
		cq->tail = cq->head;
}

void ChainQueueShow(ChainQueue *cq)
{
	assert(cq);
	CQN *q = cq->head;
	//while (q->data != NULL){//“!=”:“int”与“void *”的间接级别不同
	while (q != NULL){
		printf("%d ", q->data);
		q = q->next;
	}
	printf("\n");
}

QueueElemType ChainQueueFront(ChainQueue *cq)
{
	assert(cq);
	if (IsEmpty(cq))
		printf("队列已空\n");

	return cq->head->data;
}

QueueElemType ChainQueueBack(ChainQueue *cq)
{
	assert(cq);
	if (IsEmpty(cq))
		printf("队列已空\n");

	return cq->tail->data;
}

typedef struct {
	ChainQueue q1;
	ChainQueue q2;
} MyStack;

/** Initialize your data structure here. */

MyStack* myStackCreate() {
	MyStack *pst = (MyStack*)malloc(sizeof(MyStack));
	ChainQueueInit(&(pst->q1));
	ChainQueueInit(&(pst->q2));

	return pst;
}

/** Push element x onto stack. */
void myStackPush(MyStack* obj, int x) {
	ChainQueue *p;
	if (IsEmpty(&(obj->q1)))
		p = &(obj->q2);
	else
		p = &(obj->q1);
	ChainQueueEn(p, x);
}

/** Removes the element on top of the stack and returns that element. */
int myStackPop(MyStack* obj) {
	ChainQueue *pnoempty, *pempty;
	if (IsEmpty(&(obj->q1))){
		pempty = &(obj->q1);
		pnoempty = &(obj->q2);
	}
	else
	{
		pnoempty = &(obj->q1);
		pempty = &(obj->q2);
	}

	int val;
	while (!IsEmpty(pnoempty))
	{
		val = ChainQueueFront(pnoempty);
		ChainQueueDe(pnoempty);
		if (IsEmpty(pnoempty))
			break;
		ChainQueueEn(pempty, val);
	}
	return val;
}

/** Get the top element. */
int myStackTop(MyStack* obj) {
	ChainQueue *pnoempty, *pempty;
	if (IsEmpty(&(obj->q1)))
	{
		pempty = &(obj->q1);
		pnoempty = &(obj->q2);
	}
	else
	{
		pnoempty = &(obj->q1);
		pempty = &(obj->q2);
	}

	int val;
	while (!IsEmpty(pnoempty))
	{
		val = ChainQueueFront(pnoempty);
		ChainQueueDe(pnoempty);
		ChainQueueEn(pempty, val);
	}
	return val;
}

/** Returns whether the stack is empty. */
bool myStackEmpty(MyStack* obj) {
	return IsEmpty(&(obj->q1)) && IsEmpty(&(obj->q2));
}

void myStackFree(MyStack* obj) {
	ChainQueueDestroy(&(obj->q1));
	ChainQueueDestroy(&(obj->q2));
	free(obj);
	obj = NULL;
}

/**
* Your MyStack struct will be instantiated and called as such:
* MyStack* obj = myStackCreate();
* myStackPush(obj, x);

* int param_2 = myStackPop(obj);

* int param_3 = myStackTop(obj);

* bool param_4 = myStackEmpty(obj);

* myStackFree(obj);
*/

//232. 用栈实现队列
#define StackElemType int

typedef struct ChainStackNode
{
	StackElemType data;
	//CSN* next;//错误
	struct ChainStackNode* next;
}CSN;

typedef struct ChainStack
{
	CSN *head;
}ChainStack;

void ChainStackInit(ChainStack *cs);
void ChainDestroy(ChainStack *cs);
void ChainStackPush(ChainStack *cs, StackElemType x);
void ChainStackPop(ChainStack *cs);
void ChainStackShow(ChainStack *cs);
StackElemType ChainStackTop(ChainStack *cs);
bool ChainStackEmpty(ChainStack *cs){
	return cs->head == NULL;
}

void ChainStackInit(ChainStack *cs)
{
	cs->head = NULL;
}

void ChainStackPush(ChainStack *cs, StackElemType x)
{
	assert(cs);
	CSN *s = (CSN*)malloc(sizeof(CSN));
	assert(s);
	s->data = x;
	//if (cs->head == NULL)
	//	cs->head = s;
	//else{
	//	CSN *p = cs->head;
	//	s->next = p;
	//	cs->head = s;
	//}

	s->next = cs->head;
	cs->head = s;
}

void ChainStackPop(ChainStack *cs)
{
	assert(cs);
	if (cs->head == NULL){
		printf("栈空\n");
		return;
	}

	else{
		CSN *p = cs->head;
		cs->head = p->next;
		free(p);
	}
}

void ChainStackShow(ChainStack *cs)
{
	assert(cs);
	CSN *p = cs->head;
	while (p != NULL){
		printf("%d ", p->data);
		p = p->next;
	}
	printf("\n");
}

StackElemType ChainStackTop(ChainStack *cs)
{
	//assert(cs);
	//if (cs->head == NULL){
	//	printf("无元素可取\n");
	//	return ;
	//}//返回的值可能与元素里面的值相同
	assert(cs && cs->head);
	return cs->head->data;
}

void ChainDestroy(ChainStack *cs)
{
	assert(cs);
	//CSN *p = cs->head;
	while (cs->head != NULL){
		CSN *p = cs->head;
		cs->head = p->next;
		free(p);
	}
}

typedef struct {
	ChainStack cs1;
	ChainStack cs2;
} MyQueue;

/** Initialize your data structure here. */

MyQueue* myQueueCreate() {
	MyQueue *Seq = (MyQueue*)malloc(sizeof(MyQueue));
	ChainStackInit(&(Seq->cs1));
	ChainStackInit(&(Seq->cs2));
	return Seq;
}

/** Push element x to the back of queue. */
void myQueuePush(MyQueue* obj, int x) {
	ChainStackPush(&(obj->cs1), x);
}

/** Removes the element from in front of queue and returns that element. */
int myQueuePop(MyQueue* obj) {
	if (ChainStackEmpty(&(obj->cs2)))
	{
		while (!ChainStackEmpty(&(obj->cs1)))
		{
			ChainStackPush(&(obj->cs2), ChainStackTop(&(obj->cs1)));
			ChainStackPop(&(obj->cs1));
		}
	}
	int val = ChainStackTop(&(obj->cs2));
	ChainStackPop(&(obj->cs2));
	return val;
}

/** Get the front element. */
int myQueuePeek(MyQueue* obj) {
	if (ChainStackEmpty(&(obj->cs2)))
	{
		while (!ChainStackEmpty(&(obj->cs1)))
		{
			ChainStackPush(&(obj->cs2), ChainStackTop(&(obj->cs1)));
			ChainStackPop(&(obj->cs1));
		}
	}
	int val = ChainStackTop(&(obj->cs2));
	return val;
}

/** Returns whether the queue is empty. */
bool myQueueEmpty(MyQueue* obj) {
	return ChainStackEmpty(&(obj->cs1)) && ChainStackEmpty(&(obj->cs2));
}

void myQueueFree(MyQueue* obj) {
	ChainDestroy(&(obj->cs1));
	ChainDestroy(&(obj->cs2));
	free(obj);
	obj = NULL;
}

/**
* Your MyQueue struct will be instantiated and called as such:
* MyQueue* obj = myQueueCreate();
* myQueuePush(obj, x);

* int param_2 = myQueuePop(obj);

* int param_3 = myQueuePeek(obj);

* bool param_4 = myQueueEmpty(obj);

* myQueueFree(obj);
*/

//709. 转换成小写字母
char * toLowerCase(char * str){

	for (int i = 0, n = strlen(str); i<n; i++){

		if (*(str + i) >= 'A' && *(str + i) <= 'Z')
			*(str + i) ^= 1 << 5;//第五位的异或可以把大小写转换
	}
	return str;
}

//217. 存在重复元素
int cmp(const void*_a, const void*_b){
	int a = *(int*)_a, b = *(int*)_b;
	return a - b;
}

bool containsDuplicate(int* nums, int numsSize){
	qsort(nums, numsSize, sizeof(int), cmp);
	for (int i = 0; i<numsSize - 1; i++){
		if (nums[i] == nums[i + 1])
			return true;
	}
	return false;
}

//35. 搜索插入位置
int searchInsert(int* nums, int numsSize, int target) {
	int left = 0, right = numsSize - 1;//ans;//错误
	int ans = numsSize;//在[1,3,5,6] 7 

	while (left <= right){

		int mid = (right - left) / 2 + left;
		if (target <= nums[mid]){
			ans = mid;
			right = mid - 1;
		}
		else
			left = mid + 1;
	}
	return ans;
}

//2021_2_17
//383. 赎金信
bool canConstruct(char * ransomNote, char * magazine){

	// char count[26]={0};//错误
	int count[26] = { 0 };
	int i = 0;
	char ch = 0;

	for (i = 0; (ch = magazine[i]) != '\0'; i++)
		count[ch - 'a']++;
	for (i = 0; (ch = ransomNote[i]) != '\0'; i++){
		if (count[ch - 'a'] == 0)
			return false;
		else
			count[ch - 'a']--;
	}
	return true;
}

//58. 最后一个单词的长度
int lengthOfLastWord(char * s){
	int len = strlen(s);
	int count = 0;

	for (int i = len - 1; i >= 0; i--){
		if (s[i] != ' ')
			count++;
		if (s[i] == ' ' && 0 != count)
			break;
	}
	return count;
}

//925. 长按键入
// bool isLongPressedName(char * name, char * typed){
//     int i=0, j=0;
//     while(j<strlen(typed)){
//         if(name[i] == typed[j])
//             i++,j++;
//         else j++;
//     }
//     if(j>=strlen(typed) || i!=strlen(name)-1)
//         return false;
//     return true;
// }

bool isLongPressedName(char * name, char * typed){
	int n = strlen(name), m = strlen(typed);
	int i = 0, j = 0;
	while (j<m){

		if (i<n && name[i] == typed[j])
			i++, j++;
		else if (j>0 && typed[j] == typed[j - 1])
			j++;
		else
			return false;
	}
	return i == n;
}

//977. 有序数组的平方
/**
* Note: The returned array must be malloced, assume caller calls free().
*/

// int cmp(const void* _a, const void* _b){
//     int a = *(int*)_a, b = *(int*)_b;
//     return a-b;
// }

// int* sortedSquares(int* nums, int numsSize, int* returnSize){
//     *returnSize =numsSize;
//     int* nums1 =(int*)malloc(sizeof(int)*numsSize);
//     for(int i=0; i<numsSize; i++){
//         nums1[i]=(nums[i] * nums[i]);
//     }
//     qsort(nums1, numsSize,sizeof(int),cmp);
//     return nums1;
// }

int* sortedSquares(int* nums, int numsSize, int* returnSize){
	*returnSize = numsSize;
	int* numsnew = (int*)malloc(sizeof(int)* numsSize);//zz
	for (int i = 0, j = numsSize - 1, pos = numsSize - 1; i <= j;){

		if (nums[i] * nums[i] > nums[j] * nums[j]){
			numsnew[pos--] = nums[i] * nums[i];
			i++;
		}
		else{
			numsnew[pos--] = nums[j] * nums[j];
			j--;
		}
	}
	return numsnew;
}

//917. 仅仅反转字母
char *reverseOnlyLetters(char *s)
{
	if (s == NULL)
		return NULL;
	int i = 0, j = strlen(s) - 1;
	while (i < j){

		while (i < j && !isalpha(s[i]))
			i++;

		while (i < j && !isalpha(s[j]))
			j--;

		if (i < j){
			char tmp = s[i];
			s[i] = s[j];
			s[j] = tmp;
			i++, j--;
		}
	}
	return s;
}





//2021_2_18
//66. 加一
/**
* Note: The returned array must be malloced, assume caller calls free().
*/
int* plusOne(int* digits, int digitsSize, int* returnSize){

	int count = 0;

	for (int i = digitsSize - 1; i >= 0; i--){
		if (digits[i] == 9){
			digits[i] = 0;
			count++;
		}
		else{
			//digits[i]+=1;
			digits[i]++;
			*returnSize = digitsSize;
			return digits;
		}
	}
	int *res = (int*)malloc(sizeof(int)*(digitsSize + 1));
	memset(res, 0, sizeof(int)*(digitsSize + 1));

	if (count == digitsSize){
		res[0] = 1;
	}
	*returnSize = digitsSize + 1;
	return res;
}

//414. 第三大的数
int thirdMax(int* nums, int numsSize){
	int i;
	int first, second, third;
	int min = nums[0];
	for (i = 0; i<numsSize; i++){
		if (nums[i]<min)
			min = nums[i];
	}

	first = second = third = min;

	for (i = 0; i<numsSize; i++){

		if (nums[i] > first){
			third = second;
			second = first;
			first = nums[i];
		}
		else if (nums[i] > second && nums[i] != first){
			third = second;
			second = nums[i];
		}
		else if (nums[i] > third && nums[i] != second && nums[i] != first){
			third = nums[i];
		}
	}

	if (first == second || second == third)
		return first;
	return third;
}

//67. 二进制求和
void reverse(char *s)
{
	int n = strlen(s);
	for (int i = 0; i<n / 2; i++){
		//int tmp = s[i];
		char tmp = s[i];
		s[i] = s[n - 1 - i];
		s[n - 1 - i] = tmp;
	}
}
char * addBinary(char * a, char * b){
	reverse(a);
	reverse(b);

	int len_a = strlen(a), len_b = strlen(b);
	int n = fmax(len_a, len_b);
	char* ans = (char*)malloc(sizeof(char)*(n + 2));//进位和'\0'
	int carry = 0, len = 0;
	for (int i = 0; i<n; i++){
		carry += i < len_a ? (a[i] == '1') : 0;
		carry += i < len_b ? (b[i] == '1') : 0;
		ans[len++] = carry % 2 + '0';
		carry /= 2;
	}

	if (carry)
		ans[len++] = '1';
	ans[len] = '\0';

	reverse(ans);

	return ans;
}

//2021_2_23
//8. 字符串转换整数 (atoi)
int myAtoi(char * s){

	while (*s == ' ')
		s++;

	int flag = 1;
	if (*s == '+')
		s++;
	else if (*s == '-'){
		s++;
		flag = -1;
	}

	int ret = 0;
	int _max = INT_MAX / 10; //ret的值不能超过这个值
	while (*s <= '9' && *s >= '0'){

		int tmp = *s - '0';
		if (ret < _max || (ret == _max && tmp < 8)){

			ret = ret * 10 + tmp;
			*s++;
		}
		else
			return (flag == 1 ? INT_MAX : INT_MIN);
	}
	return flag * ret;
}

//34. 在排序数组中查找元素的第一个和最后一个位置
/**
* Note: The returned array must be malloced, assume caller calls free().
*/
int* searchRange(int* nums, int numsSize, int target, int* returnSize){

	int i, j;
	int flag = 1;
	*returnSize = 2;
	int *s = (int*)malloc(sizeof(int)* 2);
	for (i = 0; i <numsSize; i++){
		if (nums[i] == target){
			s[0] = i;
			break;
		}
	}
	for (j = numsSize - 1; j >= 0; j--){
		if (nums[j] == target){ //1个也可以
			s[1] = j;
			flag = 0;
			break;
		}
	}
	if (flag == 1){
		s[0] = -1, s[1] = -1;
	}
	return s;
}

//125. 验证回文串
// bool isPalindrome(char * s){

//     int sz =strlen(s);
//     int left = 0;
//     int right = sz-1;

//     while(left <= right){
//         if(isalnum(s[left])){//isalnum 判断是否为数字或字母
//             left++;
//         continue; //不满足条件则跳过这次循环；
//         }
//         if(isalnum(s[right])){
//             right--;
//         continue;
//         }

//         if(tolower(s[left]) != tolower(s[right]))
//             return false;
//     }
//     return true;
// }//错误，超时 ，原因:left,right可能不会变化

bool isPalindrome(char * s){

	int n = strlen(s) - 1;

	if (s == NULL)
		return false;
	if (strlen(s) == 0)
		return true;

	for (int i = 0; i <= n;){
		if (!isalnum(s[i])){//isalnum 判断是否为数字或字母
			i++;
			continue; //不满足条件则跳过这次循环；
		}
		if (!isalnum(s[n])){
			n--;
			continue;
		}

		if (tolower(s[i]) != tolower(s[n])) //tolower 字母字符统一变成小写字母
			return false;
		//更新条件
		i++, n--;
	}
	return true;
}

//443. 压缩字符串
int compress(char* chars, int charsSize){

int cur = 0;

for (int i = 0, j = 0; i < charsSize; j = i){

	while (i < charsSize && chars[i] == chars[j])
		i++;
	chars[cur++] = chars[j];
	if (i - j == 1)
		continue;

	char s[1000];
	sprintf(s, "%d", i - j);

	for (int z = 0; z<strlen(s); z++)
		chars[cur++] = s[z];
}
return cur;
}

//581. 最短无序连续子数组
// int capare(const void *a, const void *b){
//     return *(int*)a-*(int*)b;}
// int findUnsortedSubarray(int* nums, int numsSize){
//     int begin = 0, end = 0;
//     int *numscpy = (int*)malloc(numsSize*sizeof(int));
//     //memcpy(numscpy, nums, numsSize);
//     memcpy(numscpy, nums, sizeof(int)*numsSize);
//     qsort(numscpy,numsSize,sizeof(int),capare);
//     for(int i =0; i < numsSize; i++){
//         if(numscpy[i] != nums[i]){
//             begin = i;
//             break;
//         }
//     }
//     for(int j =numsSize-1; j>=0; j--){
//         if(numscpy[j] != nums[j]){
//             end = j;
//             break;
//         }
//     }
//     if(begin == end)
//         return 0;
//     return end-begin+1;
// }

int findUnsortedSubarray(int* nums, int numsSize){
	int start = 0;
	//int end = 0;//错误
	int end = -1;
	int max = nums[0];
	int min = nums[numsSize - 1];
	for (int i = 0; i < numsSize; i++){
		if (nums[i]<max)
			end = i;
		else
			max = nums[i];

		if (nums[numsSize - 1 - i]>min)
			start = numsSize - 1 - i;
		else
			min = nums[numsSize - 1 - i];
	}
	return end - start + 1;
}



//2021_2_24
//150. 逆波兰表达式求值
int chartoint(char *str)
{
	int i = str[0] == '-' ? 1 : 0, num = 0;
	while (str[i])
		num = num * 10 + str[i++] - '0';
	return str[0] == '-' ? -num : num;
}

int evalRPN(char ** tokens, int tokensSize){
	int i = 1, stack_index = 0;
	int stack[(tokensSize + 1) / 2];
	stack[0] = chartoint(tokens[0]);
	while (i<tokensSize){
		switch (tokens[i++][0]){
		case'+':
			stack[--stack_index] = stack[stack_index - 1] + stack[stack_index];
			break;
		case'*':
			stack[--stack_index] = stack[stack_index - 1] * stack[stack_index];
			break;
		case'/':
			stack[--stack_index] = stack[stack_index - 1] / stack[stack_index];
			break;
		case'-': //判定是否是符号
			if (tokens[i - 1][1] == 0){//
				stack[--stack_index] = stack[stack_index - 1] - stack[stack_index];
				break;
			}
		default:
			stack[++stack_index] = chartoint(tokens[i - 1]);
		}
	}
	return stack[0];
}

//78. 子集
/**
* Return an array of arrays of size *returnSize.
* The sizes of the arrays are returned as *returnColumnSizes array.
* Note: Both returned array and *columnSizes array must be malloced, assume caller calls free().
*/
int** subsets(int* nums, int numsSize, int* returnSize, int** returnColumnSizes){
	// int **ans = (int*)malloc(sizeof(int) * (1 << numsSize));
	int **ans = malloc(sizeof(int*)* (1 << numsSize)); //
	*returnColumnSizes = malloc(sizeof(int)* (1 << numsSize));
	*returnSize = 1 << numsSize;
	int t[numsSize];
	for (int mask = 0; mask< (1 << numsSize); mask++){
		int tSize = 0;
		for (int i = 0; i < numsSize; i++){
			if (mask & (1 << i)){
				t[tSize++] = nums[i];
			}
		}
		int *tmp = malloc(sizeof(int)*tSize);
		memcpy(tmp, t, sizeof(int)* tSize);
		(*returnColumnSizes)[mask] = tSize;//??
		ans[mask] = tmp;
	}
	return ans;
}

//199二叉树的右视图
void travel(struct TreeNode* root, int depth, int* returnSize, int* res){
	if (root == NULL)
		return;
	if (depth > *returnSize){
		res[*returnSize] = root->val;
		*returnSize = depth;
	}
	travel(root->right, depth + 1, returnSize, res);
	travel(root->left, depth + 1, returnSize, res);
}


int* rightSideView(struct TreeNode* root, int* returnSize){
	*returnSize = 0;
	int *res = (int*)malloc(sizeof(int)* 1000);
	travel(root, 1, returnSize, res);
	return res;
}

//2021_2_27
//1688. 比赛中的配对次数
int numberOfMatches(int n){
	int sum = 0;
	while (n > 1){
		if ((n % 2) == 0){
			sum += n / 2;
			n /= 2;
		}
		else{
			sum += (n - 1) / 2;
			n = (n - 1) / 2 + 1;
		}
	}
	return sum;
}

int numberOfMatches(int n){
	int sum = 0;
	while (n > 1){
		sum += n / 2;
		n = (n + 1) / 2;
	}
	return sum;
}

int numberOfMatches(int n){
	return n - 1;
}

//53. 最大子序和
int maxSubArray(int* nums, int numsSize){
	int pre = 0, max = nums[0];
	for (int i = 0; i < numsSize; i++){
		pre = fmax(pre + nums[i], nums[i]);
		max = fmax(pre, max);
	}
	return max;
}

//392. 判断子序列
bool isSubsequence(char * s, char * t){
	int s_len = strlen(s);
	int t_len = strlen(t);

	int i = 0, j = 0;
	while (i < s_len && j < t_len){
		if (s[i] == t[j])
			i++;
		j++;
	}
	// if(i == s_len)
	//     return true;
	// return false;
	return i == s_len;
}

//2021_2_28
//896. 单调数列
// bool isMonotonic(int* A, int ASize){

//     if(ASize == 0)
//         return true;

//     int add = 1, sub =1;
//     for(int i = 0; i < ASize-1; i++){

//         if(A[i] <= A[i+1])
//             add++;
//         if(A[i] >= A[i+1])
//             sub++;
//     }

//     if(add == ASize || sub == ASize)
//         return true;
//     return false;
// }

// bool isMonotonic(int* A, int ASize){
//     bool inc = true;
//     bool dec = true;

//     for(int i = 0; i < ASize -1; i++){
//         if(A[i] < A[i+1])
//             dec = false;
//         if(A[i] > A[i+1])
//             inc = false;
//     }
//     return inc || dec;
// }
bool isMonotonic(int* A, int ASize){
	int i = 0;
	if (A[0] > A[ASize - 1]){
		for (i = 1; i < ASize; i++)
		if (A[i] > A[i - 1]) return false;
	}
	else if (A[0] < A[ASize - 1]){
		for (i = 1; i < ASize; i++)
		if (A[i] < A[i - 1]) return false;
	}
	else{
		for (i = 1; i<ASize; i++)
		if (A[i] != A[i - 1]) return false;
	}
	return true;
}

//110. 平衡二叉树
/**
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     struct TreeNode *left;
*     struct TreeNode *right;
* };
*/

int _maxdepth(struct TreeNode* root){
	if (root == NULL)
		return 0;
	int left = _maxdepth(root->left);
	int right = _maxdepth(root->right);

	return (left > right ? left : right) + 1;
}

bool isBalanced(struct TreeNode* root){
	if (root == NULL)
		return true;
	int l_depth = _maxdepth(root->left);
	int r_depth = _maxdepth(root->right);

	return abs(l_depth - r_depth) < 2 && isBalanced(root->left) && isBalanced(root->right);
}

//897. 递增顺序查找树
/**
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     struct TreeNode *left;
*     struct TreeNode *right;
* };
*/

#define MAX 1000

struct TreeNode* increasingBST(struct TreeNode* root){
	struct TreeNode* ret = (struct TreeNode*)malloc(sizeof(struct TreeNode));
	struct TreeNode* p = ret;
	struct TreeNode* stack[MAX];

	int top = -1;
	while (root != NULL || top != -1){

		while (root != NULL){
			stack[++top] = root;
			root = root->left;
		}
		if (top != -1){

			root = stack[top--];
			p->right = (struct TreeNode*)malloc(sizeof(struct TreeNode));
			p = p->right;
			p->val = root->val;
			p->left = NULL;
			p->right = NULL;
			root = root->right;
		}
	}
	return ret->right;
}

//剑指 Offer 54. 二叉搜索树的第k大节点
/**
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     struct TreeNode *left;
*     struct TreeNode *right;
* };
*/
#define MAX 10000

int* inorder(struct TreeNode* root, int *returnsize)
{
	//struct TreeNode* stack[MAX];
	struct TreeNode **stack = (struct TreeNode**)malloc(sizeof(struct TreeNode*) * 1000);
	int *res = (int*)malloc(sizeof(int)*MAX);
	int top = -1;
	*returnsize = 0;
	struct TreeNode *p = root;
	while (p != NULL || top != -1){
		if (p != NULL){
			stack[++top] = p;
			p = p->left;
		}
		else{
			p = stack[top--];
			res[(*returnsize)++] = p->val;
			p = p->right;
		}
	}
	return res;
}

int kthLargest(struct TreeNode* root, int k){
	int size = 0;
	int *tmp = inorder(root, &size);
	return tmp[size - k];
}

int target_index;
struct TreeNode *targetNode;

void kthLargestCore(struct TreeNode*node)
{
	if (node == NULL)
	{
		return;
	}
	kthLargestCore(node->right);
	// if(target_index == 0)//找到后，剩下的就停止
	// {
	//     return;
	// }//对效率是否有影响
	target_index--;
	if (target_index == 0)
	{
		targetNode = node;
		return;
	}
	kthLargestCore(node->left);
}

int kthLargest(struct TreeNode* root, int k){
	target_index = k;
	kthLargestCore(root);
	return targetNode->val;
}

//112. 路径总和
/**
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     struct TreeNode *left;
*     struct TreeNode *right;
* };
*/


bool hasPathSum(struct TreeNode* root, int targetSum){
	if (root == NULL)
		return false;
	if (root->left == NULL && root->right == NULL)
		return targetSum == root->val;
	return hasPathSum(root->left, targetSum - root->val) || hasPathSum(root->right, targetSum - root->val);
}

//1022. 从根到叶的二进制数之和
/**
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     struct TreeNode *left;
*     struct TreeNode *right;
* };
*/
//int num = 0;//这样写的话，无法再次利用，num的值没有刷新
int num;
void SRL(struct TreeNode* root, int sum)
{
	if (root == NULL)
		return;

	sum = (sum << 1) + root->val;
	if (root->left == NULL && root->right == NULL){
		num += sum;
	}
	SRL(root->left, sum);
	SRL(root->right, sum);

}

int sumRootToLeaf(struct TreeNode* root){
	int sum = 0;
	num = 0;
	SRL(root, sum);
	return num;
}

//617. 合并二叉树
/**
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     struct TreeNode *left;
*     struct TreeNode *right;
* };
*/


struct TreeNode* mergeTrees(struct TreeNode* root1, struct TreeNode* root2){
	if (root1 == NULL)
		return root2;
	if (root2 == NULL)
		return root1;

	struct TreeNode *merge = (struct TreeNode*)malloc(sizeof(struct TreeNode));
	merge->val = root1->val + root2->val;
	merge->left = mergeTrees(root1->left, root2->left);
	merge->right = mergeTrees(root1->right, root2->right);

	return merge;
}

//剑指 Offer 55 - I. 二叉树的深度
/**
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     struct TreeNode *left;
*     struct TreeNode *right;
* };
*/


int maxDepth(struct TreeNode* root){
	if (root == NULL)
		return 0;
	int L_Depth = maxDepth(root->left);
	int R_Depth = maxDepth(root->right);

	return (L_Depth > R_Depth ? L_Depth : R_Depth) + 1;
}

//2021_3_1
//303. 区域和检索 - 数组不可变
typedef struct {
	int *sum;
} NumArray;


NumArray* numArrayCreate(int* nums, int numsSize) {
	NumArray *ret = malloc(sizeof(NumArray));
	ret->sum = malloc(sizeof(NumArray)*(numsSize + 1));
	ret->sum[0] = 0;
	for (int i = 0; i < numsSize; i++){
		ret->sum[i + 1] = ret->sum[i] + nums[i];
	}
	return ret;

}

int numArraySumRange(NumArray* obj, int i, int j) {
	return obj->sum[j + 1] - obj->sum[i];
}

void numArrayFree(NumArray* obj) {
	free(obj);//free(obj->sum);两者有何异同？
}

/**
* Your NumArray struct will be instantiated and called as such:
* NumArray* obj = numArrayCreate(nums, numsSize);
* int param_1 = numArraySumRange(obj, i, j);

* numArrayFree(obj);
*/

//867. 转置矩阵
/**
* Return an array of arrays of size *returnSize.
* The sizes of the arrays are returned as *returnColumnSizes array.
* Note: Both returned array and *columnSizes array must be malloced, assume caller calls free().
*/
int** transpose(int** matrix, int matrixSize, int* matrixColSize, int* returnSize, int** returnColumnSizes){
	int row = matrixColSize[0];
	int col = matrixSize;
	int **tmp = (int**)malloc(sizeof(int*)* row); //int*类型
	*returnSize = row; //记录数组元素的个数
	*returnColumnSizes = malloc(sizeof(int)*row);
	for (int i = 0; i < row; i++){
		tmp[i] = malloc(sizeof(int)*col);
		(*returnColumnSizes)[i] = col;//记录数组元素中元素的个数
	}
	for (int i = 0; i< row; i++){
		for (int j = 0; j < col; j++){
			tmp[i][j] = matrix[j][i];
		}
	}
	return tmp;
}

//面试题 17.10. 主要元素
int majorityElement(int* nums, int numsSize){
	int res = nums[0], count = 1;
	for (int i = 1; i < numsSize; i++){
		if (nums[i] == res){
			count++;
			continue;
		}
		if (--count == 0){
			count = 1;
			res = nums[i];
		}
	}
	if (numsSize <= 2){
		return count == numsSize ? res : -1;
	}
	// if(count >= (numsSize+1)/2)
	//     return res;
	if (count > numsSize / 2)
		retunr res;

	count = 0;
	for (int i = 0; i < numsSize; i++){
		if (res == nums[i])
			count++;
	}

	if (count >= (numsSize + 1) / 2)
		return res;
	return -1;
}

//977. 有序数组的平方
/**
* Note: The returned array must be malloced, assume caller calls free().
*/
// int cmpare(const void *_a, const void *_b){
//     int a = *(int*)_a, b = *(int*)_b;
//     return a-b;
// }
// int* sortedSquares(int* nums, int numsSize, int* returnSize){

//     *returnSize = numsSize;
//     for(int i = 0; i < numsSize; i++)
//         nums[i] = nums[i] * nums[i];
//     qsort(nums, numsSize, sizeof(int), cmpare);
//     return nums;//新数组不清楚是什么意思？新创一个数组的话：改变原来的数组
// }

int* sortedSquares(int* nums, int numsSize, int* returnSize){
	*returnSize = numsSize;
	int *numsNew = (int*)malloc(sizeof(int)*numsSize);
	//for(int i=0,j=numsSize-1,pos=numsSize-1;i<j;){//[-4,-1,0,3,10]中0的计算会被遗漏//奇偶都不行
	for (int i = 0, j = numsSize - 1, pos = numsSize - 1; i <= j;){
		if (nums[i] * nums[i] > nums[j] * nums[j]){
			numsNew[pos--] = nums[i] * nums[i];
			i++;
		}
		else{
			numsNew[pos--] = nums[j] * nums[j];
			j--;
		}
	}
	return numsNew;
}


// int* sortedSquares(int* nums, int numsSize, int* returnSize){
//     *returnSize = numsSize;
//     for(int i = 0; i < numsSize; i++)
//         nums[i] = nums[i] * nums[i];
//     int tmp = nums[0];
//     for(int i = 0; i < numsSize; i++){
//         if(tmp > nums[i]){
//             nums[0] = nums[i];
//             nums[i] = tmp;
//             break;
//         }
//     }
//     return nums;
// }//不要运行//题目要求不能有局部递减存在这种方法不适合这里

//1185. 一周中的第几天
char * dayOfTheWeek(int day, int month, int year){
	int flag = 0, sum = 0;;
	int dif[][12] =
	{
		{ 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 },
		{ 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 }
	};
	// for(int i = 1971 ;i < year; i++){
	//     if((i%4 == 0 && i%100 !=0) || (i%400 == 0))
	//         sum+=2;//错了很多次了，else不写的话，if判定成功还是要在sum+=1一次
	//     sum+=1;
	// }
	for (int i = 1971; i < year; i++){
		if ((i % 4 == 0 && i % 100 != 0) || (i % 400 == 0))
			sum += 2;
		else
			sum += 1;
	}


	if ((year % 4 == 0 && year % 100 != 0) || (year % 400 == 0))
		flag = 1;

	for (int i = 0; i <month - 1; i++)
		sum += dif[flag][i];

	sum += day;
	sum--;//

	int data = (5 + (sum % 7)) % 7;
	switch (data){
	case 0:return "Sunday";
	case 1:return "Monday";
	case 2:return "Tuesday";
	case 3:return "Wednesday";
	case 4:return "Thursday";
	case 5:return "Friday";
	case 6:return "Saturday";
	default:
		return 0;
	}

//628. 三个数的最大乘积
	// int cmpare(int *a, int *b){
	//     return *a-*b;
	// }
	// int maximumProduct(int* nums, int numsSize){
	//     qsort(nums, numsSize, sizeof(int), cmpare);
	//     return fmax(nums[0]*nums[1]*nums[numsSize-1],nums[numsSize-3]*nums[numsSize-2]*nums[numsSize-1]);
	// }

	int maximumProduct(int* nums, int numsSize){
		int min1 = INT_MAX, min2 = INT_MAX;
		int max1 = INT_MIN, max2 = INT_MIN, max3 = INT_MIN;

		int tmp = 0;
		for (int i = 0; i < numsSize; i++){
			tmp = nums[i];
			if (tmp > max1){
				max3 = max2;
				max2 = max1;
				max1 = tmp;
			}
			else if (tmp > max2){
				max3 = max2;
				max2 = tmp;
			}
			else if (tmp > max3){
				max3 = tmp;
			}

			if (tmp < min1){
				min2 = min1;
				min1 = tmp;
			}
			else if (tmp < min2)
				min2 = tmp;
		}
		return fmax(min1*min2*max1, max1*max2*max3);
	}

//190. 颠倒二进制位
//uint32_t reverseBits(uint32_t n) {

	uint32_t ans = 0;
	for (int i = 0; i < 32; i++){
		ans = (ans << 1) + (1 & n);
		n = n >> 1;
	}
	return ans;
}

//268. 丢失的数字
//// int cmp(const void*_a, const void*_b){
//     int a = *(int*)_a, b = *(int*)_b;
//     return a - b;
// }
// int missingNumber(int* nums, int numsSize){
//     qsort(nums,numsSize,sizeof(int),cmp);
//     for(int i = 0; i < numsSize; i++){
//         if(nums[i] != i)
//             return i;
//     }
//     return numsSize;
// }29.37%,5.82%

// int missingNumber(int* nums, int numsSize){
//     int res = numsSize;
//     for(int i = 0; i < numsSize; i++){
//         res^=nums[i]^i;
//     }
//     return res;
// }

// int missingNumber(int* nums, int numsSize){
//     int sum = numsSize;
//     for(int i = 0; i < numsSize; i++){
//         sum+=i-nums[i];
//     }
//     return sum;
// }

//2021_3_2
//989. 数组形式的整数加法
/**
* Note: The returned array must be malloced, assume caller calls free().
*/
// int* addToArrayForm(int* A, int ASize, int K, int* returnSize){
//     int *res = (int*)malloc(sizeof(int)*fmax(100,ASize+1));
//     *returnSize = 0;
//     int sum = 0;
//     for(int i = ASize - 1; i >= 0; --i){

//         if(K > 0){
//         sum = A[i] + K%10;
//         K/=10;
//         if(sum > 9){
//             K++;
//             sum-=10;
//             }
//         res[(*returnSize)++] = sum;
//         }
//         else
//             res[(*returnSize)++] =A[i];
//     }
//     for(;K>0;K/=10)
//         res[(*returnSize)++] = K%10;
//     for(int i=0;i<(*returnSize)/2;i++){
//         int tmp=res[i];
//         res[i]=res[(*returnSize)-1-i];
//         res[(*returnSize)-1-i]=tmp;
//     }
//     return res;
// }
int* addToArrayForm(int* A, int ASize, int K, int* returnSize){
	int *res = (int*)malloc(sizeof(int)*fmax(100, ASize + 1));
	*returnSize = 0;
	for (int i = ASize - 1; i >= 0 || K>0; i--, K /= 10){
		if (i >= 0)
			K += A[i];
		res[(*returnSize)++] = K % 10;
	}
	int tmp;
	for (int i = 0; i < (*returnSize) / 2; i++){
		tmp = res[i];
		res[i] = res[(*returnSize) - 1 - i];
		res[(*returnSize) - 1 - i] = tmp;
	}
	return res;
}

//304. 二维区域和检索 - 矩阵不可变
typedef struct {
	int **sum;
	int sumSize;
} NumMatrix;


NumMatrix* numMatrixCreate(int** matrix, int matrixSize, int* matrixColSize) {
	NumMatrix *ret = malloc(sizeof(NumMatrix));
	ret->sum = malloc(sizeof(int*)*matrixSize);
	ret->sumSize = matrixSize;
	for (int i = 0; i < ret->sumSize; i++){
		ret->sum[i] = malloc(sizeof(int)*(matrixColSize[i] + 1));
		ret->sum[i][0] = 0;
		for (int j = 0; j < matrixColSize[i]; j++)
			ret->sum[i][j + 1] = ret->sum[i][j] + matrix[i][j];
	}
	return ret;
}

int numMatrixSumRegion(NumMatrix* obj, int row1, int col1, int row2, int col2) {
	int sum = 0;
	for (int i = row1; i <= row2; i++)
		sum += obj->sum[i][col2 + 1] - obj->sum[i][col1];
	return sum;
}

void numMatrixFree(NumMatrix* obj) {
	//free(obj);
	for (int i = 0; i < obj->sumSize; i++)
		free(obj->sum[i]);
}

/**
* Your NumMatrix struct will be instantiated and called as such:
* NumMatrix* obj = numMatrixCreate(matrix, matrixSize, matrixColSize);
* int param_1 = numMatrixSumRegion(obj, row1, col1, row2, col2);

* numMatrixFree(obj);
*/

//剑指 Offer 57. 和为s的两个数字
/**
* Note: The returned array must be malloced, assume caller calls free().
*/
int* twoSum(int* nums, int numsSize, int target, int* returnSize){
	*returnSize = 2;
	int *ret = (int*)malloc(sizeof(int)*(*returnSize));
	int left = 0;
	int right = numsSize - 1;
	while (left < right){
		if (nums[left] + nums[right] == target){
			ret[0] = nums[left];
			ret[1] = nums[right];
			return ret;
		}
		else if (nums[left] + nums[right] > target)
			right--;
		else  left++;
	}
	return NULL;
}

//剑指 Offer 53 - II. 0～n-1中缺失的数字
// int missingNumber(int* nums, int numsSize){
//     for(int i = 0; i < numsSize; i++){
//         if(nums[i] != i)
//             return i;
//     }
//     return numsSize;
// }

// int missingNumber(int* nums, int numsSize){
//     for(int i = 0; i < numsSize-1; i++){
//         if(nums[i+1] -nums[i] > 1)
//             return i+1;
//     }
//     if(nums[0] == 0){
//     return numsSize;
//     }
//     return 0;
// 

int missingNumber(int* nums, int numsSize){
	int left = 0, right = numsSize - 1;
	int mid;
	while (left <= right){
		mid = left + (right - left) / 2;
		if (nums[mid] > mid)
			right = mid - 1;
		else left = mid + 1;
	}
	return left;
}

//面试题 10.05. 稀疏数组搜索
int findString(char** words, int wordsSize, char* s){
	for (int i = 0; i < wordsSize; i++){
		if (!strcmp(words[i], s))
			return i;
	}
	return -1;
}

//350. 两个数组的交集 II
/**
* Note: The returned array must be malloced, assume caller calls free().
*/
// int cmp(const void*_a,const void*_b){
//     int a = *(int*)_a, b = *(int*)_b;
//     return a - b;
// }[-2147483648,1,2,3]
//  [1,-2147483648,-2147483648] //越界了
int cmp(const void*_a, const void*_b){
	int a = *(int*)_a, b = *(int*)_b;
	return a == b ? 0 : a > b ? 1 : -1;//计算可能导致计算出范围
}
int* intersect(int* nums1, int nums1Size, int* nums2, int nums2Size, int* returnSize){
	qsort(nums1, nums1Size, sizeof(int), cmp);
	qsort(nums2, nums2Size, sizeof(int), cmp);

	int retSize = nums1Size > nums2Size ? nums2Size : nums1Size;
	int *ret = (int*)malloc(sizeof(int)*retSize);
	*returnSize = 0;
	for (int index1 = 0, index2 = 0; index1 < nums1Size && index2 < nums2Size;){
		if (nums1[index1] > nums2[index2]){
			index2++;
		}
		else if (nums1[index1] < nums2[index2]){
			index1++;
		}
		else{
			ret[(*returnSize)++] = nums1[index1];
			index1++, index2++;
		}
	}
	return ret;
}

//69. x 的平方根
// int mySqrt(int x){
//     int y = 0;
//     if(x == 1)
//         return 1;
//     if(x == 0)
//         return 0;
//     for(y = 0; y < x; y++){
//         if(y*y < x && (y+1)*(y+1)>x )
//             break;
//         if(y*y == x)
//             break;
//     }
//     return y;
// }//垃圾


// int mySqrt(int x){
//     int l = 0 ,h = x, mid;
//     while(l<=h){
//         mid = l+(h-l)/2;
//         if((long long)mid*mid == x)
//             return mid;
//         else if ((long long)mid*mid < x)
//             l = mid + 1;
//         else 
//             h = mid - 1;
//     } 
//     return h;
// }

// int mySqrt(int x){
//     int L = 1 ,R = x/2 + 1;
//     int mid;
//     while(L <= R){
//         mid = L + (R-L)/2;
//         if(mid > x/mid){
//             R = mid - 1;
//         }
//         else if(mid  < x/mid){
//             L = mid + 1;
//         }
//         else return mid;
//     }
//     return R;
// }
int mySqrt(int x){
	if (x == 0)
		return 0;
	if (x == 1)
		return 1;
	int L = 1, R = x / 2 + 1;
	int mid;
	while (L <= R){
		mid = L + (R - L) / 2;
		if (mid > x / mid){
			R = mid - 1;
		}
		else if (mid  < x / mid){
			L = mid + 1;
		}
		else return mid;
	}
	return R;
}

//剑指 Offer 11. 旋转数组的最小数字
// int cmp(const void*_a, const void*_b){
//     int a = *(int*)_a, b = *(int*)_b;
//     return a == b ? 0 : a > b ? 1 : -1;
// }

// int minArray(int* numbers, int numbersSize){
//     qsort(numbers, numbersSize,sizeof(int),cmp);
//     return numbers[0];
// }

int minArray(int* numbers, int numbersSize){
	int left = 0, right = numbersSize - 1;
	int mid;
	while (left < right){
		mid = (left >> 1) + (right >> 1);
		if (numbers[mid] > numbers[right])
			left = mid + 1;
		else if (numbers[mid] < numbers[left]){
			right = mid;
		}
		else
			right--;
	}
	return numbers[right];
}

//167. 两数之和 II - 输入有序数组/**
*Note: The returned array must be malloced, assume caller calls free().
* /
int* twoSum(int* numbers, int numbersSize, int target, int* returnSize){
	*returnSize = 2;
	int *ret = (int*)malloc(sizeof(int)*(*returnSize));
	int left = 0, right = numbersSize - 1;
	while (left <= right){
		if (numbers[left] + numbers[right] > target)
			right--;
		else if (numbers[left] + numbers[right] < target)
			left++;
		else break;
	}
	ret[0] = left + 1;
	ret[1] = right + 1;
	return ret;
}

//441. 排列硬币
// int arrangeCoins(int n){
//     int res = 0;
//     for(int i = 1; i<=n; i++){
//         res++;
//         n-=i;
//     }
//     return res;
// }

// int arrangeCoins(int n){
//     long sum = 0;
//     int i = 0;
//     while(sum<=n) sum+=++i;
//     return i-1;
// }

int arrangeCoins(int n){
	int left = 1, right = n;
	long sum = 0;
	//int mid;//？？
	long mid;
	while (left <= right){
		mid = left + (right - left) / 2;
		sum = mid*(mid + 1) / 2;
		if (sum > n) right = mid - 1;
		else if (sum < n) left = mid + 1;
		else return mid;
	}
	return right;
}

//744. 寻找比目标字母大的最小字母
// char nextGreatestLetter(char* letters, int lettersSize, char target){
//     int count = 0;
//     int i;
//     for(i = 0; i < lettersSize; i++){
//         if((letters[i] - 'a') > (target - 'a'))
//             break;
//         count++;
//     }
//     if(count == lettersSize)
//         return  letters[0];
//     return letters[i];
// }
char nextGreatestLetter(char* letters, int lettersSize, char target){
	int left = 0, right = lettersSize - 1;
	if (letters[right] <= target)
		return letters[0];

	int mid;
	while (left < right){
		mid = (left >> 1) + (right >> 1);
		if (letters[mid] > target)
			right = mid;
		else if (letters[mid] <= target)
			left = mid + 1;
	}
	return letters[right];
}

//852. 山脉数组的峰顶索引
// int peakIndexInMountainArray(int* arr, int arrSize){
//     int left = 0, right = arrSize-1;
//     int mid;
//     while(left <= right){
//         mid = left + (right -left)/2;
//         if(arr[mid] > arr[mid+1] && arr[mid] > arr[mid-1])
//             return mid;
//         else if(arr[mid] < arr[mid+1])
//             left = mid + 1;
//         else right = mid - 1;
//     }
//     return -1;
// }

int peakIndexInMountainArray(int* arr, int arrSize){
	int left = 0, right = arrSize - 1;
	int mid;
	while (left <right){
		mid = left + (right - left) / 2;
		if (arr[mid] > arr[mid + 1] && arr[mid] > arr[mid - 1])
			return mid;
		else if (arr[mid] < arr[mid + 1])
			left = mid + 1;
		else right = mid - 1;
	}
	return right;
}

//1351. 统计有序矩阵中的负数
// int countNegatives(int** grid, int gridSize, int* gridColSize){
//     int count = 0;
//     int i , j;
//     for(i =0; i < gridSize; i++){
//         for(j = 0; j < gridColSize[i]; j++){
//             if(grid[i][j] < 0)
//                 count++;
//         }
//     }
//     return count;
// }1.0

// int countNegatives(int** grid, int gridSize, int* gridColSize){
//     int count = 0;
//     int i , j;
//     for(i =0; i < gridSize; i++){
//         for(j = gridColSize[i]-1; j >= 0; j--){
//             if(grid[i][j] < 0)
//                 count++;
//             else   
//                 break;
//         }
//     }
//     return count;
// }2.0

int countNegatives(int** grid, int gridSize, int* gridColSize){

	int i, j, mid, sum = 0;
	for (i = 0; i < gridSize; i++){
		int left = 0, right = gridColSize[i];
		while (left < right){
			mid = left + (right - left) / 2;
			if (grid[i][mid] >= 0){
				left = mid + 1;
			}
			else
				right = mid;
		}
		sum += gridColSize[i] - right;
	}
	return sum;
}

//374. 猜数字大小
/**
* Forward declaration of guess API.
* @param  num   your guess
* @return 	     -1 if num is lower than the guess number
*			      1 if num is higher than the guess number
*               otherwise return 0
* int guess(int num);
*/

int guessNumber(int n){
	int left = 1, right = n;
	int mid;
	while (left <= right){
		mid = left + (right - left) / 2;
		if (guess(mid)<0)
			right = mid - 1;
		else if (guess(mid) == 0)
			return mid;
		else
			left = mid + 1;
	}
	return right;
}

//2021_3_3
//338. 比特位计数
/**
* Note: The returned array must be malloced, assume caller calls free().
*/
// int* countBits(int num, int* returnSize){
//     int *sum = (int*)malloc(sizeof(int)*(num+1));
//     *returnSize = 0;
//     for(int i = 0; i <= num; i++){
//         int count = 0;
//         int tmp = i;
//         while(tmp!=0){
//             if(tmp & 1 == 1)
//                 count++;
//             tmp=tmp>>1;
//         }
//         sum[(*returnSize)++] = count;
//     }
//     return sum;
// }

// int countOnes(int x){
//     int count = 0;
//     while(x > 0){
//         x = x & (x - 1);
//         count++;
//     }
//     return count;
// }

// int* countBits(int num, int* returnSize){
//     int *sum = (int*)malloc(sizeof(int)*(num + 1));
//     *returnSize = num + 1;
//     for(int i = 0; i <= num; i++){
//         sum[i] = countOnes(i);
//     }
//     return sum;
// }

// int* countBits(int num, int* returnSize){
//     int *sum = (int*)malloc(sizeof(int)*(num + 1));
//     *returnSize = num + 1;
//     sum[0] = 0;
//     int highbt = 0;
//     for(int i = 1; i <= num; i++){
//         //if(i & (i-1) == 0)
//         if((i & (i-1)) == 0)
//             highbt = i;
//         sum[i] = sum[i - highbt] + 1;
//     }
//     return sum;
// }

// int* countBits(int num, int* returnSize){
//     int *sum = (int*)malloc(sizeof(int)*(num+1));
//     *returnSize = num + 1;
//     sum[0] = 0;
//     int lowbt = 0;
//     for(int i = 1; i <= num; i++){
//         sum[i] = sum[i>>1]+(i&1);
//     }
//     return sum;
// }

int* countBits(int num, int* returnSize){
	int *sum = (int*)malloc(sizeof(int)*(num + 1));
	*returnSize = num + 1;
	sum[0] = 0;
	for (int i = 1; i <= num; i++){
		sum[i] = sum[i &(i - 1)] + 1;
	}
	return sum;
}

//349. 两个数组的交集
/**
* Note: The returned array must be malloced, assume caller calls free().
*/
int cmp(const void*_a, const void*_b){
	int a = *(int*)_a, b = *(int*)_b;
	return a == b ? 0 : a > b ? 1 : -1;
}
int* intersection(int* nums1, int nums1Size, int* nums2, int nums2Size, int* returnSize){
	qsort(nums1, nums1Size, sizeof(int), cmp);
	qsort(nums2, nums2Size, sizeof(int), cmp);
	int index1 = 0, index2 = 0;
	int *sum = (int*)malloc(sizeof(int)*(nums1Size + nums2Size));
	*returnSize = 0;
	while (index1 < nums1Size && index2 < nums2Size){
		int num1 = nums1[index1], num2 = nums2[index2];
		if (num1 == num2){
			if (!(*returnSize) || num1 != sum[(*returnSize) - 1]){
				sum[(*returnSize)++] = num1;
				//index1++,index2++;
			}
			index1++, index2++;
		}
		else if (num1 < num2)
			index1++;
		else index2++;
	}
	return sum;
}

//367. 有效的完全平方数
bool isPerfectSquare(int num){
	long x = 0;//int x 为何会溢出 //Char 35:运行时错误：有符号整数溢出：46341*46341不能在类型“int”[solution.c]中表示
	while (x <= (num / 2 + 1)){
		if (x*x == num)
			return true;
		else if (x*x < num && (x + 1)*(x + 1) > num)
			break;
		else if (x*x < num)
			x++;
	}
	return false;
}

//278. 第一个错误的版本
// The API isBadVersion is defined for you.
// bool isBadVersion(int version);

int firstBadVersion(int n) {
	int left = 0, right = n;
	while (left < right){
		int mid = (left >> 1) + (right >> 1);
		if (isBadVersion(mid) == true)
			right = mid;
		else if (isBadVersion(mid) == false)
			left = mid + 1;
	}
	return left;
}

//704. 二分查找
int search(int* nums, int numsSize, int target){
	int left = 0, right = numsSize - 1;
	while (left <= right){
		int mid = left + (right - left) / 2;
		if (nums[mid] < target)
			left = mid + 1;
		else if (nums[mid] > target)
			right = mid - 1;
		else
			return mid;
	}
	return -1;
}

//剑指 Offer 53 - I. 在排序数组中查找数字 I
int searchleft(int* nums, int numsSize, int target){
	int left = 0, right = numsSize - 1;
	int mid;
	while (left <= right){
		mid = left + (right - left) / 2;
		if (nums[mid] >= target)
			right = mid - 1;
		else
			left = mid + 1;
	}
	return left;
}

int searchright(int* nums, int numsSize, int target){
	int left = 0, right = numsSize - 1;
	int mid;
	while (left <= right){
		mid = left + (right - left) / 2;
		if (nums[mid] <= target)
			left = mid + 1;
		else
			right = mid - 1;
	}
	return right;
}

int search(int* nums, int numsSize, int target){
	if (numsSize == 0)
		return 0;
	int left = searchleft(nums, numsSize, target);
	if (left < 0)
		return 0;
	int right = searchright(nums, numsSize, target);
	return right - left + 1;
}

//74. 搜索二维矩阵
//// bool searchMatrix(int** matrix, int matrixSize, int* matrixColSize, int target){
//     for(int i = 0; i < matrixSize; i++){
//         if(matrix[i][(*matrixColSize)] > target){
//             int left = 0, right = (*matrixColSize) - 1;
//             while(left <= right){
//                 int mid = left + (right - left)/2;
//                 if(matrix[i][mid] > target)
//                     right = mid - 1;
//                 else if(matrix[i][mid] < target)
//                     left = mid + 1;
//                 else  return true;
//             }
//         }
//     }
//     return false;
// }
// bool searchMatrix(int** matrix, int matrixSize, int* matrixColSize, int target){
//     int left = 0, right = matrixSize*(*matrixColSize) - 1;
//     int mid, row, col;
//     while(left <= right){
//         mid = left + (right - left)/2;
//         row = mid / *matrixColSize;
//         col = mid % *matrixColSize;
//         if(matrix[row][col] < target)
//             left = mid + 1;
//         else if(matrix[row][col] > target)
//             right = mid - 1;
//         else return true;
//     }
//     return false;
// }

bool searchMatrix(int** matrix, int matrixSize, int* matrixColSize, int target){
	int left = 0, right = matrixSize - 1, mid_row, mid;
	bool refresh = false;
	while (left <= right){
		mid_row = left + (right - left) / 2;
		if (matrix[mid_row][0] > target){
			right = mid_row - 1;
			refresh = true;
		}
		if (matrix[mid_row][*matrixColSize - 1] < target){
			left = mid_row + 1;
			refresh = true;
		}
		if (!refresh){
			left = 0, right = *matrixColSize - 1;
			while (left <= right){
				mid = left + (right - left) / 2;
				if (matrix[mid_row][mid] > target)
					right = mid - 1;
				else if (matrix[mid_row][mid] < target)
					left = mid + 1;
				else return true;
			}
			break;
		}
		else refresh = false;
	}
	return false;
}

//275. H 指数 II
int hIndex(int* citations, int citationsSize){
	int left = 0, right = citationsSize, mid;
	while (left < right){
		mid = (left + right + 1) / 2;
		if (citations[citationsSize - mid] >= mid)
			left = mid;
		else
			right = mid - 1;
	}
	return left;
}

//441. 排列硬币
// int arrangeCoins(int n){
//     int res = 0;
//     for(int i = 1; i<=n; i++){
//         res++;
//         n-=i;
//     }
//     return res;
// }

// int arrangeCoins(int n){
//     long sum = 0;
//     int i = 0;
//     while(sum<=n) sum+=++i;
//     return i-1;
// }

int arrangeCoins(int n){
	int left = 1, right = n;
	long sum = 0;
	//int mid;//？？
	int mid;
	while (left <= right){
		mid = left + (right - left) / 2;
		sum = mid*(mid + 1) / 2;
		if (sum > n) right = mid - 1;
		else if (sum < n) left = mid + 1;
		else return mid;
	}
	return right;
}

//540. 有序数组中的单一元素
// int singleNonDuplicate(int* nums, int numsSize){
//     int res = 0;
//     for(int i = 0; i < numsSize; i++)
//         res^=nums[i];
//     return res;
// }O(n);

int singleNonDuplicate(int* nums, int numsSize){
	if (numsSize == 1)
		return nums[0];
	int left = 0, right = numsSize - 1, mid;
	while (left <= right){
		mid = left + (right - left) / 2;
		if (mid == 0 && nums[mid] != nums[mid + 1])
			return nums[mid];
		else if (mid == numsSize - 1 && nums[mid] != nums[mid - 1])
			return nums[mid];
		else if (nums[mid] != nums[mid + 1] && nums[mid] != nums[mid - 1])
			return nums[mid];

		if (mid % 2 != 0){
			if (nums[mid] == nums[mid - 1])
				left = mid + 1;
			else right = mid - 1;
		}
		if (mid % 2 == 0){
			if (nums[mid] == nums[mid + 1])
				left = mid + 1;
			else right = mid - 1;
		}
	}
	return -1;
}

//378. 有序矩阵中第 K 小的元素
// int cmp(const void*_a, const void*_b){
//     int a = *(int*)_a, b = *(int*)_b;
//     return a == b ? 0 : a > b ? 1 : -1;
// }

// int kthSmallest(int** matrix, int matrixSize, int* matrixColSize, int k){
//     int *dst = (int*)malloc(sizeof(int)*(matrixSize*(*matrixColSize)));
//     int returnSize = 0;
//     for(int i = 0 ; i < matrixSize; i++){
//         for(int j = 0; j < (*matrixColSize); j++){
//            dst[returnSize++] = matrix[i][j];
//         }
//     }
//     qsort(dst,returnSize,sizeof(int),cmp);
//     return dst[k-1];
// }
bool check(int** matrix, int mid, int k, int n){
	int i = n - 1;
	int j = 0;
	int num = 0;
	while (i >= 0 && j<n){
		if (matrix[i][j] <= mid){
			num += i + 1;
			j++;
		}
		else i--;
	}
	return num >= k;
}

int kthSmallest(int** matrix, int matrixSize, int* matrixColSize, int k){
	int left = matrix[0][0], right = matrix[matrixSize - 1][matrixSize - 1];
	while (left < right){
		int mid = left + (right - left) / 2;
		if (check(matrix, mid, k, matrixSize)){
			right = mid;
		}
		else  left = mid + 1;
	}
	return left;
}

//354. 俄罗斯套娃信封问题
int cmp(const int**a, const int**b){
	return (*a)[0] == (*b)[0] ? (*b)[1] - (*a)[1] : (*a)[0] - (*b)[0];
}

int maxEnvelopes(int** envelopes, int envelopesSize, int* envelopesColSize){
	if (envelopesSize == 0)
		return 0;
	qsort(envelopes, envelopesSize, sizeof(int*), cmp);
	int i, n = envelopesSize;
	int f[n];
	for (i = 0; i < n; i++)
		f[i] = 1;
	int ret = 1;
	for (i = 1; i < n; i++){
		for (int j = 0; j < i; j++){
			if (envelopes[j][1] < envelopes[i][1])
				f[i] = fmax(f[i], f[j] + 1);
		}
		ret = fmax(ret, f[i]);
	}
	return ret;
}

//1365. 有多少小于当前数字的数字
/**
* Note: The returned array must be malloced, assume caller calls free().
*/
// int* smallerNumbersThanCurrent(int* nums, int numsSize, int* returnSize){
//     //int count[101] = {0};
//     int count[101];
//     memset(count,0,sizeof(count));
//     int i;
//     for(i = 0; i < numsSize; i++)
//         count[nums[i]]++;
//     for(i = 1; i <= 100; i++)
//         count[i]+=count[i-1];

//     for(i = 0; i < numsSize; i++){
//         nums[i] = nums[i] == 0 ? 0 : count[nums[i]-1];
//     }
//     *returnSize = numsSize;
//     return nums;
// }哈希

// int* smallerNumbersThanCurrent(int* nums, int numsSize, int* returnSize){
//     int *news = (int*)malloc(sizeof(int)*numsSize);
//     for(int i = 0; i < numsSize; i++){
//         int count = 0;
//         for(int j = 0; j <numsSize; j++){
//             if(nums[j] < nums[i])
//                 count++ ;
//         }
//         news[i] = count;
//     }
//     *returnSize = numsSize;
//     return news;
// }暴力破解

int cmp(const int*a, const int *b){
	return *a - *b;
}

int* smallerNumbersThanCurrent(int* nums, int numsSize, int* returnSize){
	qsort(nums, numsSize, sizeof(int), cmp);
	int *news = (int*)malloc(sizeof(int)*numsSize);
	news[0] = 0;
	for (int i = 1; i < numsSize; i++){
		int j = i - 1;
		while (nums[i] == nums[j] && j >= 0)
			j--;
		news[i] = j + 1;
	}
	return news;
}

//463. 岛屿的周长
// const int dx[4] = { -1,  0, 0, 1};
// const int dy[4] = {  0, -1, 1, 0};

// int islandPerimeter(int** grid, int gridSize, int* gridColSize){
//     int count = 0;
//     for(int i = 0; i < gridSize; i++){
//         for(int j = 0; j < gridColSize[0]; j++){
//             if(grid[i][j]){
//                 for(int k = 0; k <4; k++){
//                     int tx = dx[k] + i;
//                     int ty = dy[k] + j;
//                 if(tx < 0 || ty < 0 || tx >= gridSize || ty >= gridColSize[0] || !grid[tx][ty])
//                     count+=1;
//             }
//         }
//     }
// }
//     return count;
// }

// int islandPerimeter(int** grid, int gridSize, int* gridColSize){
// }

//1748. 唯一元素的和
int sumOfUnique(int* nums, int numsSize){
	//int count[101] = {0};
	int count[101];
	memset(count, 0, sizeof(count));
	for (int i = 0; i < numsSize; i++){
		count[nums[i]]++;
	}
	int ret = 0;
	for (int j = 0; j < numsSize; j++){
		if (count[nums[j]] == 1)
			ret += nums[j];
	}
	return ret;
}

//136. 只出现一次的数字
// int singleNumber(int* nums, int numsSize){
//     int ret = 0;
//     for(int i = 0; i < numsSize; ++i){
//         ret^=nums[i];
//     }
//     return ret;
// }

// int singleNumber(int* nums, int numsSize){
//     int hash[1001] = {0};
//     for(int i = 0; i < numsSize; i++){
//         hash[nums[i]]++;//Line 4: Char 13: runtime error: index -1 out of bounds for type 'int [1001]' [solution.c]//无法解决负数的保存问题
//     }
//     int count = 0;
//     for(int i = 0; i < numsSize; i++){
//         if(hash[nums[i]] == 1)
//             return nums[i];
//     }
//     return -1;
// }//错误示例

// int singleNumber(int* nums, int numsSize){
//     int max = INT_MIN, min = INT_MAX;
//     for(int i = 0; i < numsSize; i++){
//         max = fmax(nums[i],max);
//         min = fmin(nums[i],min);
//     }
//     int len = max - min + 1;
//     //int hash[len + 1] = {0};//可变大小的对象可能未初始化[solution.c] 为什么？
//     int *hash = (int*)malloc(sizeof(int)*len);
//     memset(hash,0,sizeof(int)*len);
//     for(int i = 0; i < numsSize; i++){
//         hash[nums[i]- min]++;
//     }
//     for(int i = 0; i < numsSize; i++){
//         if(hash[nums[i]-min] == 1)
//             return nums[i];
//     }
//     return -1;
// }

//1. 两数之和
class Solution {
public:
	vector<int> twoSum(vector<int>& nums, int target) {
		int n = nums.size();
		for (int i = 0; i < n; i++){
			for (int j = i + 1; j < n; j++){
				if (nums[i] + nums[j] == target)
					return{ i, j };
			}
		}
		return{};
	}
};

//1437. 是否所有 1 都至少相隔 k 个元素
class Solution {
public:
	bool kLengthApart(vector<int>& nums, int k) {
		int n = nums.size();
		int prev = -1;
		for (int i = 0; i < n; i++){
			if (nums[i] == 1){
				if (prev != -1 && i - prev - 1<k)
					return false;
				prev = i;
			}
		}
		return true;
	}
};

//1047. 删除字符串中的所有相邻重复项
char * removeDuplicates(char * S){
	int n = strlen(S);
	char *stack = (char*)malloc(sizeof(char)*(n + 1));
	int RTZ = 0;
	for (int i = 0; i < n; i++){
		if (RTZ > 0 && stack[RTZ - 1] == S[i]){
			RTZ--;
		}
		else
			stack[RTZ++] = S[i];
	}
	stack[RTZ] = '\0';
	return stack;
}


int main()
{
	EXIT_SUCCESS;
}