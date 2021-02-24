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

int main()
{
	EXIT_SUCCESS;
}