#include<iostream>
using namespace std;

//2021_3_29

//40. 组合总和 II
class Solution {
private:
	vector<vector<int>> result;
	vector<int> path;
	void Trackbreaking(vector<int>& candidates, int target, int sum, int Startindex, vector<bool>& used) {
		if (sum == target) {
			result.push_back(path);
			return;
		}
		for (int i = Startindex; i < candidates.size() && sum + candidates[i] <= target; ++i) {

			if (i > 0 && candidates[i] == candidates[i - 1] && used[i - 1] == false) {
				continue;
			}
			sum += candidates[i];
			path.push_back(candidates[i]);
			used[i] = true;
			Trackbreaking(candidates, target, sum, i + 1, used);
			used[i] = false;
			sum -= candidates[i];
			path.pop_back();
		}
	}
public:
	vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {

		vector<bool> used(candidates.size(), false);

		result.clear();
		path.clear();
		sort(candidates.begin(), candidates.end());
		Trackbreaking(candidates, target, 0, 0, used);

		return result;
	}
};

//2021_3_28

//39. 组合总和
class Solution {
public:
	vector<vector<int>> result;
	vector<int> tmp;

	void Trackbreaking(vector<int>& candidates, int target, int sum, int Startindex) {

		// if (sum > target) {
		//     return;
		// }

		if (sum == target) {
			result.push_back(tmp);
			return;
		}

		for (int i = Startindex; i < candidates.size() && sum + candidates[i] <= target; i++) {
			tmp.push_back(candidates[i]);
			sum += candidates[i];
			Trackbreaking(candidates, target, sum, i);//这个Startindex加不加的区分
			sum -= candidates[i];
			tmp.pop_back();
		}
	}

	vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
		result.clear();
		tmp.clear();
		if (candidates.size() == 0) return result;
		sort(candidates.begin(), candidates.end());
		Trackbreaking(candidates, target, 0, 0);

		return result;
	}
};

// class Solution {
// public:
//     vector<vector<int>> result;
//     vector<int> tmp;

//     void Trackbreaking(vector<int>& candidates, int target, int sum) {

//         if (sum > target) {
//             return;
//         }

//         if (sum == target) {
//             result.push_back(tmp);
//             return;
//         }

//         for (int i = 0; i < candidates.size(); i++) {
//             tmp.push_back(candidates[i]);
//             sum += candidates[i];
//             Trackbreaking(candidates, target, sum);
//             sum -= candidates[i];
//             tmp.pop_back();
//         }
//     }

//     vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
//         result.clear();
//         tmp.clear();
//         if (candidates.size() == 0) return result;
//         Trackbreaking(candidates, target, 0);

//         return result;
//     }
// };

//17. 电话号码的字母组合
class Solution {
public:
	const string leetermap[10] {
		"", //0
			"", //1
			"abc",  //2
			"def",  //3
			"ghi",  //4
			"jkl",  //5
			"mno",  //6
			"pqrs", //7
			"tuv",  //8
			"wxyz", //9
	};
	vector<string> result;
	string s;
	void Trackbreaking(const string& digits, int index) {

		if (index == digits.size()) {
			result.push_back(s);
			return;
		}

		int digit = digits[index] - '0';
		string leeters = leetermap[digit];

		for (int i = 0; i < leeters.size(); i++) {
			s.push_back(leeters[i]);
			Trackbreaking(digits, index + 1);
			s.pop_back();
		}
	}

	vector<string> letterCombinations(string digits) {
		result.clear();
		s.clear();

		if (digits.size() == 0) return result;
		Trackbreaking(digits, 0);

		return result;
	}
};

//216. 组合总和 III
class Solution {
public:
	vector<vector<int>> result;
	vector<int> path;
	void Trackbreaking(int k, int Startindex, int sum, int Targetnum) {

		if (sum > Targetnum) return;

		if (path.size() == k) {
			if (sum == Targetnum) result.push_back(path);
			return;
		}

		for (int i = Startindex; i <= 9; i++) {
			path.push_back(i);
			sum += i;
			Trackbreaking(k, i + 1, sum, Targetnum);
			sum -= i;
			path.pop_back();
		}
	}
	vector<vector<int>> combinationSum3(int k, int n) {
		result.clear();
		path.clear();
		Trackbreaking(k, 1, 0, n);

		return result;
	}
};

//77. 组合
class Solution {
public:
	vector<vector<int>> result;
	vector<int> path;
	void TrachBreaking(int n, int k, int startIndex) {

		if (path.size() == k) {
			result.push_back(path);
			return;
		}

		for (int i = startIndex; i <= n - (k - path.size()) + 1; i++) { //举个例子去看判定条件
			path.push_back(i);
			TrachBreaking(n, k, i + 1);
			path.pop_back();
		}
	}
	vector<vector<int>> combine(int n, int k) {
		result.clear();
		path.clear();
		TrachBreaking(n, k, 1);
		return result;
	}
};

//538. 把二叉搜索树转换为累加树
/**
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     TreeNode *left;
*     TreeNode *right;
*     TreeNode() : val(0), left(nullptr), right(nullptr) {}
*     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
*     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
* };
*/
class Solution {
public:
	TreeNode* convertBST(TreeNode* root) {
		stack<TreeNode*> st;
		TreeNode *cur = root;
		int pre = 0;

		while (cur != nullptr || !st.empty()) {

			if (cur != nullptr) {
				st.push(cur);
				cur = cur->right;
			}
			else {
				cur = st.top(); st.pop();
				cur->val += pre;
				pre = cur->val;
				cur = cur->left;
			}
		}
		return root;
	}
};

//108. 将有序数组转换为二叉搜索树
/**
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     TreeNode *left;
*     TreeNode *right;
*     TreeNode() : val(0), left(nullptr), right(nullptr) {}
*     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
*     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
* };
*/
class Solution {
public:
	TreeNode *Travelsal(vector<int>& nums, int left, int right) {
		if (left > right) return nullptr;
		int mid = left + (right - left) / 2;

		TreeNode *root = new TreeNode(nums[mid]);

		root->left = Travelsal(nums, left, mid - 1);
		root->right = Travelsal(nums, mid + 1, right);

		return root;
	}
	TreeNode* sortedArrayToBST(vector<int>& nums) {
		TreeNode *root = Travelsal(nums, 0, nums.size() - 1);
		return root;
	}
};

//2021_3_27

//669. 修剪二叉搜索树
/**
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     TreeNode *left;
*     TreeNode *right;
*     TreeNode() : val(0), left(nullptr), right(nullptr) {}
*     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
*     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
* };
*/
class Solution {
public:
	TreeNode* trimBST(TreeNode* root, int low, int high) {
		if (root == NULL)  return NULL;

		if (root->val < low) {
			TreeNode *right = trimBST(root->right, low, high);
			return right;
		}
		if (root->val > high) {
			TreeNode *left = trimBST(root->left, low, high);
			return left;
		}
		root->left = trimBST(root->left, low, high);
		root->right = trimBST(root->right, low, high);

		return root;
	}
};

//450. 删除二叉搜索树中的节点
/**
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     TreeNode *left;
*     TreeNode *right;
*     TreeNode() : val(0), left(nullptr), right(nullptr) {}
*     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
*     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
* };
*/
class Solution {
public:
	TreeNode* deleteNode(TreeNode* root, int key) {
		if (root == NULL) return root;
		if (root->val == key) {
			if (root->left == NULL) return root->right;
			else if (root->right == NULL) return root->left;
			else {
				TreeNode *cur = root->right;
				while (cur->left != NULL) {
					cur = cur->left;
				}
				cur->left = root->left;
				TreeNode *tmp = root;
				root = root->right;
				delete tmp;
				return root;
			}
		}
		if (root->val > key) root->left = deleteNode(root->left, key);
		if (root->val < key) root->right = deleteNode(root->right, key);

		return root;
	}
};

//701. 二叉搜索树中的插入操作
/**
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     TreeNode *left;
*     TreeNode *right;
*     TreeNode() : val(0), left(nullptr), right(nullptr) {}
*     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
*     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
* };
*/
class Solution {
public:
	TreeNode* insertIntoBST(TreeNode* root, int val) {
		if (root == NULL) {
			TreeNode *node = new TreeNode(val);
			return node;
		}
		TreeNode *cur = root;
		TreeNode *parent = root;

		while (cur != NULL) {
			parent = cur;
			if (cur->val < val) cur = cur->right;
			else cur = cur->left;
		}
		TreeNode *node = new TreeNode(val);
		if (parent->val > val) parent->left = node;
		else parent->right = node;

		return root;
	}
};

//236. 二叉树的最近公共祖先
/**
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     TreeNode *left;
*     TreeNode *right;
*     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
* };
*/
class Solution {
public:
	TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
		if (root == p || root == q || root == NULL) return root;

		TreeNode *left = lowestCommonAncestor(root->left, p, q);
		TreeNode *right = lowestCommonAncestor(root->right, p, q);

		if (left != NULL && right != NULL) return root;
		else if (left == NULL && right != NULL) return right;
		else if (left != NULL && right == NULL) return left;
		else return NULL;
	}
};

//501. 二叉搜索树中的众数
*     TreeNode *right;
*TreeNode() : val(0), left(nullptr), right(nullptr) {}
*TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
*TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
*};
*/
class Solution {
public:
	vector<int> findMode(TreeNode* root) {
		stack<TreeNode*> st;
		vector<int> result;
		TreeNode *pre = NULL;
		TreeNode *cur = root;
		int maxcount = 0;
		int count = 0;

		while (cur != NULL || !st.empty()) {

			if (cur != NULL) {
				st.push(cur);
				cur = cur->left;
			}
			else{
				cur = st.top(); st.pop();
				if (pre == NULL) {
					count = 1;
				} //else if (pre != NULL || pre->val == cur->val) { //傻子
				else if (pre->val == cur->val) {
					count++;
				}
				else {
					count = 1;
				}
				if (count == maxcount) {
					result.push_back(cur->val);
				}

				if (count > maxcount) {
					maxcount = count;
					result.clear();
					result.push_back(cur->val);
				}
				pre = cur;
				cur = cur->right;
			}
		}
		return result;
	}
};

//530. 二叉搜索树的最小绝对差
/**
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     TreeNode *left;
*     TreeNode *right;
*     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
* };
*/
class Solution {
public:
	int getMinimumDifference(TreeNode* root) {
		stack<TreeNode*> st;
		TreeNode *cur = root;
		TreeNode *pre = NULL;
		int result = INT_MAX;

		while (cur != NULL || !st.empty()) {

			if (cur != NULL) {
				st.push(cur);
				cur = cur->left;
			}
			else {
				cur = st.top(); st.pop();

				if (pre != NULL) {
					result = min(result, cur->val - pre->val);
				}

				pre = cur;
				cur = cur->right;
				//st.push(cur);
			}
		}
		return result;
	}
};

//61. 旋转链表
/**
* Definition for singly-linked list.
* struct ListNode {
*     int val;
*     ListNode *next;
*     ListNode() : val(0), next(nullptr) {}
*     ListNode(int x) : val(x), next(nullptr) {}
*     ListNode(int x, ListNode *next) : val(x), next(next) {}
* };
*/
class Solution {
public:
	ListNode* rotateRight(ListNode* head, int k) {
		if (k == 0 || head == NULL || head->next == NULL)
			return head;
		int len = 1;
		ListNode *cur = head;
		while (cur->next) {
			cur = cur->next;
			len++;
		}
		int index = len - k%len;
		if (index == len)   return head;
		cur->next = head;
		while (index--) {
			cur = cur->next;
		}
		ListNode *newcur = cur->next;
		cur->next = NULL;
		return newcur;
	}
};

//2021_3_26

//98. 验证二叉搜索树
/**
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     TreeNode *left;
*     TreeNode *right;
*     TreeNode() : val(0), left(nullptr), right(nullptr) {}
*     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
*     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
* };
*/

class Solution {
public:
	TreeNode* pre = NULL;
	bool isValidBST(TreeNode* root) {//中序遍历
		if (root == NULL) return true;

		bool left = isValidBST(root->left);

		if (pre != NULL && pre->val >= root->val) return false;

		pre = root;

		bool right = isValidBST(root->right);

		return left && right;
	}
};

class Solution {
public:
	bool isValidBST(TreeNode* root) {
		stack<TreeNode*> st;
		TreeNode *cur = root;
		TreeNode *pre = NULL;
		//st.push(root);//错误

		while (cur != NULL || !st.empty()) {

			if (cur) {
				st.push(cur);
				cur = cur->left;
			}
			else {
				cur = st.top(); st.pop();
				//if (pre != NULL && cur->val >= cur->right->val) return false;
				if (pre != NULL && cur->val <= pre->val) return false;
				pre = cur;//很重要
				cur = cur->right;
			}
		}
		return true;
	}
};

//700. 二叉搜索树中的搜索
/**
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     TreeNode *left;
*     TreeNode *right;
*     TreeNode() : val(0), left(nullptr), right(nullptr) {}
*     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
*     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
* };
*/
class Solution {
public:
	TreeNode* searchBST(TreeNode* root, int val) {
		if (root == NULL && root->val == val) return root;
		queue<TreeNode*> que;
		que.push(root);

		while (!que.empty()) {
			TreeNode *cur = que.front(); que.pop();
			if (cur == NULL || cur->val == val) return cur;

			if (cur->val > val) que.push(cur->left);
			if (cur->val < val) que.push(cur->right);
		}
		return NULL;
	}
};

class Solution {
public:
	TreeNode* searchBST(TreeNode* root, int val) {
		while (root != NULL) {//不会用！就别用
			if (root->val > val) root = root->left;
			else if (root->val < val) root = root->right;
			else return root;
		}
		return NULL;
	}
};


//617. 合并二叉树
/**
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     TreeNode *left;
*     TreeNode *right;
*     TreeNode() : val(0), left(nullptr), right(nullptr) {}
*     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
*     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
* };
*/
class Solution {
public:
	TreeNode* mergeTrees(TreeNode* root1, TreeNode* root2) {
		if (root1 == NULL) return root2;
		if (root2 == NULL) return root1;

		TreeNode *merge = new TreeNode(0);
		merge->val = root1->val + root2->val;
		merge->left = mergeTrees(root1->left, root2->left);
		merge->right = mergeTrees(root1->right, root2->right);

		return merge;
	}
};

class Solution {
public:
	TreeNode* mergeTrees(TreeNode* root1, TreeNode* root2) {
		if (root1 == NULL) return root2;
		if (root2 == NULL) return root1;
		queue<TreeNode*> que;
		que.push(root1);
		que.push(root2);

		while (!que.empty()) {
			TreeNode* node1 = que.front(); que.pop();
			TreeNode* node2 = que.front(); que.pop();
			node1->val += node2->val;

			if (node1->left != NULL && node2->left != NULL) {
				que.push(node1->left);
				que.push(node2->left);
			}

			if (node1->right != NULL && node2->right != NULL) {
				que.push(node1->right);
				que.push(node2->right);
			}

			if (node1->left == NULL && node2->left != NULL) {
				node1->left = node2->left;
			}

			if (node1->right == NULL && node2->right != NULL) {
				node1->right = node2->right;
			}
		}
		return root1;
	}
};

//654. 最大二叉树
/**
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     TreeNode *left;
*     TreeNode *right;
*     TreeNode() : val(0), left(nullptr), right(nullptr) {}
*     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
*     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
* };
*/
class Solution {
public:
	TreeNode* Travel(vector<int>&nums, int left, int right) {
		if (left >= right) return NULL;

		int maxindex = left;
		//for (int i = left + 1; i < nums.size(); i++) {
		for (int i = left + 1; i < right; i++) {
			if (nums[i] > nums[maxindex]) maxindex = i;
		}

		TreeNode *root = new TreeNode(nums[maxindex]);

		root->left = Travel(nums, left, maxindex);
		root->right = Travel(nums, maxindex + 1, right);

		return root;
	}
	TreeNode* constructMaximumBinaryTree(vector<int>& nums) {
		return Travel(nums, 0, nums.size());
	}
};

//2021_3_25

//105. 从前序与中序遍历序列构造二叉树
/**
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     TreeNode *left;
*     TreeNode *right;
*     TreeNode() : val(0), left(nullptr), right(nullptr) {}
*     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
*     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
* };
*/
class Solution {
public:
	TreeNode* Travel(vector<int>& preorder, int prebegin, int preend, vector<int>& inorder, int inbegin, int inend) {
		if (preend == prebegin) return NULL;

		int rootValue = preorder[prebegin];
		TreeNode *root = new TreeNode(rootValue);

		if (preend - prebegin == 1) return root;

		int range;
		for (range = inbegin; range < inend; range++) {
			if (rootValue == inorder[range]) break;
		}

		int inleftbegin = inbegin;
		//int inleftend = inbegin + range;//错误
		//int inrightbegin = inbegin + range;
		int leftend = range;
		//int inrightbegin = range;错误
		int inrightbegin = range + 1;
		int inrightend = inend;

		int preleftbegin = prebegin + 1;
		int preleftend = prebegin + 1 + range - inbegin;
		int prerightbegin = prebegin + 1 + range - inbegin;
		int prerightend = preend;

		root->left = Travel(preorder, preleftbegin, preleftend, inorder, inbegin, inend);
		root->right = Travel(preorder, prerightbegin, prerightend, inorder, inrightbegin, inrightend);

		return root;
	}
	TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
		if (!preorder.size() || !inorder.size()) return NULL;
		return Travel(preorder, 0, preorder.size(), inorder, 0, inorder.size());
	}
};

//106. 从中序与后序遍历序列构造二叉树
/**
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     TreeNode *left;
*     TreeNode *right;
*     TreeNode() : val(0), left(nullptr), right(nullptr) {}
*     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
*     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
* };
*/
class Solution {
public:
	TreeNode* Travel(vector<int>& inorder, int inorderbegin, int inorderend, vector<int>& postorder, int postorderbegin, int postorderend) {
		if (postorderbegin == postorderend) return NULL;

		int rootValue = postorder[postorderend - 1];
		TreeNode *root = new TreeNode(rootValue);

		if (postorderend - postorderbegin == 1) return root;

		int range;
		for (range = inorderbegin; range < inorderend; range++) {
			if (inorder[range] == rootValue) break;
		}

		int inleftbegin = inorderbegin;
		int inleftend = range; //左闭右开
		int inrighbegin = range + 1;
		int inrightend = inorderend;

		postorder.resize(postorder.size() - 1);

		int postleftbegin = postorderbegin;
		int postleftend = postorderbegin + range - inorderbegin;
		int postrightbegin = postorderbegin + range - inorderbegin;
		int postrightend = postorderend - 1;

		root->left = Travel(inorder, inleftbegin, inleftend, postorder, postleftbegin, postleftend);
		root->right = Travel(inorder, inrighbegin, inrightend, postorder, postrightbegin, postrightend);

		return root;

	}
	TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) {
		if (!inorder.size() || !postorder.size()) return NULL;
		return Travel(inorder, 0, inorder.size(), postorder, 0, postorder.size());
	}
};

//82. 删除排序链表中的重复元素 II
/**
* Definition for singly-linked list.
* struct ListNode {
*     int val;
*     ListNode *next;
*     ListNode() : val(0), next(nullptr) {}
*     ListNode(int x) : val(x), next(nullptr) {}
*     ListNode(int x, ListNode *next) : val(x), next(next) {}
* };
*/
class Solution {
public:
	ListNode* deleteDuplicates(ListNode* head) {
		if (head == NULL || head->next == NULL) {
			return head;
		}
		ListNode *dummy = new ListNode(0, head);
		ListNode *cur = dummy;

		while (cur->next && cur->next->next) {//有cur->next->next的这种操作，都记得判定一下cur->next->next是否为空
			//while (cur->next) {//空的节点是不可能有值的

			if (cur->next->val == cur->next->next->val) {//这一步可知

				int record = cur->next->val;

				while (cur->next && cur->next->val == record) {
					cur->next = cur->next->next;
				}
			}
			else {
				cur = cur->next;
			}
		}
		return dummy->next;
	}
};

//剑指 Offer 18. 删除链表的节点
/**
* Definition for singly-linked list.
* struct ListNode {
*     int val;
*     ListNode *next;
*     ListNode(int x) : val(x), next(NULL) {}
* };
*/
class Solution {
public:
	ListNode* deleteNode(ListNode* head, int val) {
		ListNode *dummy = new ListNode(0, head);
		ListNode *cur = dummy;

		while (cur->next) {

			if (cur->next->val == val) {
				ListNode *temp = cur->next;
				cur->next = cur->next->next;
				temp->next == NULL;
			}
			else {
				cur = cur->next;
			}
		}
		return dummy->next;
	}
};

//83. 删除排序链表中的重复元素
/**
* Definition for singly-linked list.
* struct ListNode {
*     int val;
*     ListNode *next;
*     ListNode() : val(0), next(nullptr) {}
*     ListNode(int x) : val(x), next(nullptr) {}
*     ListNode(int x, ListNode *next) : val(x), next(next) {}
* };
*/
class Solution {
public:
	ListNode* deleteDuplicates(ListNode* head) {
		if (head == NULL || head->next == NULL) {
			return head;
		}
		ListNode *cur = head;

		while (cur != NULL && cur->next != NULL) {

			if (cur->val == cur->next->val) {
				ListNode *temp = cur->next;
				cur->next = cur->next->next;
				temp->next = NULL; //防止野指针
			}
			else {
				cur = cur->next;
			}
		}
		return head;
	}
};

//2021_3_24

//112. 路径总和
/**
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     TreeNode *left;
*     TreeNode *right;
*     TreeNode() : val(0), left(nullptr), right(nullptr) {}
*     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
*     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
* };
*/
class Solution {
public:
	bool Travelsl(TreeNode *cur, int count) {
		if (!cur->left && !cur->right && count == 0) return true;
		if (!cur->left && !cur->right) return false;

		if (cur->left) {
			count -= cur->left->val;
			if (Travelsl(cur->left, count)) return true;
			count += cur->left->val;
		}

		if (cur->right) {
			count -= cur->right->val;
			if (Travelsl(cur->right, count)) return true;
			count += cur->right->val;
		}

		return false;

	}
	bool hasPathSum(TreeNode* root, int targetSum) {
		if (root == NULL) return false;

		return Travelsl(root, targetSum - root->val);
	}
};

//2021_3_23

//513. 找树左下角的值
/**
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     TreeNode *left;
*     TreeNode *right;
*     TreeNode() : val(0), left(nullptr), right(nullptr) {}
*     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
*     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
* };
*/

class Solution {
public:
	int findBottomLeftValue(TreeNode* root) {//层序遍历yyds
		queue<TreeNode*> que;
		int result = 0;
		que.push(root);

		while (!que.empty()) {

			int size = que.size();

			for (int i = 0; i < size; i++) {
				TreeNode *cur = que.front();
				que.pop();
				if (i == 0) result = cur->val;
				if (cur->left) que.push(cur->left);
				if (cur->right) que.push(cur->right);
			}
		}
		return result;
	}
};

//404. 左叶子之和
/**
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     TreeNode *left;
*     TreeNode *right;
*     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
* };
*/
class Solution {
public:
	int sumOfLeftLeaves(TreeNode* root) {
		stack<TreeNode*> st;
		if (root == nullptr) return 0;
		st.push(root);

		int sum = 0;
		while (!st.empty()) {
			TreeNode *node = st.top();
			st.pop();

			if (node->right) st.push(node->right);
			if (node->left) st.push(node->left);

			if (node->left && !node->left->left && !node->left->right) {
				sum += node->left->val;
			}
		}
		return sum;
	}
};

//100. 相同的树
/**
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     TreeNode *left;
*     TreeNode *right;
*     TreeNode() : val(0), left(nullptr), right(nullptr) {}
*     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
*     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
* };
*/
class Solution {
public:
	bool isSameTree(TreeNode* p, TreeNode* q) {
		if (p == nullptr && q == nullptr) return true;
		else if (p != nullptr && q == nullptr) return false;
		else if (p == nullptr && q != nullptr) return false;
		else if (p->val != q->val) return false;
		else {
			bool left = isSameTree(p->left, q->left);
			bool right = isSameTree(p->right, q->right);
			return left && right;
		}
	}
};

//2021_3_22

//191. 位1的个数
// class Solution {
// public:
//     int hammingWeight(uint32_t n) {
//         int count = 0;
//         int total = 0;
//         while (count <= 31) {
//             if (n & 1) {
//                 total++;
//             }
//             n = n >> 1;
//             count++;
//         }
//         return total;
//     }
// };

 class Solution {
 public:
     int hammingWeight(uint32_t n) {
         if (n == 0) return 0;
         int count = 1;//循环会少一次
         while (n =(n & (n-1))) count++;
         return count;
     }
 };

//222. 完全二叉树的节点个数
/**
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     TreeNode *left;
*     TreeNode *right;
*     TreeNode() : val(0), left(nullptr), right(nullptr) {}
*     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
*     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
* };
*/

class Solution {
public:
	int getNum(TreeNode* cur) {
		if (cur == 0) return 0;
		int left_num = getNum(cur->left);
		int right_num = getNum(cur->right);
		int tree_num = left_num + right_num + 1;

		return tree_num;
	}
	int countNodes(TreeNode* root) {
		return getNum(root);
	}
};

// class Solution {
// public:
//     int countNodes(TreeNode* root) {
//         if (root == NULL) return 0;
//         queue<TreeNode*> que;
//         que.push(root);

//         int sum = 1;
//         while (!que.empty()) {
//             int size = que.size();

//             for (int i = 0; i < size; i++) {

//                 TreeNode* node = que.front();
//                 que.pop();

//                 if (node->left) {
//                     sum++;
//                     que.push(node->left);
//                 }
//                 if (node->right) {
//                     sum++;
//                     que.push(node->right);
//                 }
//             }
//         }
//         return sum;
//     }
// };

//111. 二叉树的最小深度
/**
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     TreeNode *left;
*     TreeNode *right;
*     TreeNode() : val(0), left(nullptr), right(nullptr) {}
*     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
*     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
* };
*/

class Solution {
public:
	int getDepth(TreeNode* node) {
		if (node == NULL) return 0;
		int leftDepth = getDepth(node->left);
		int rightDepth = getDepth(node->right);

		if (node->left == NULL && node->right != NULL) {
			return 1 + rightDepth;
		}
		if (node->left != NULL && node->right == NULL) {
			return 1 + leftDepth;
		}
		int result = 1 + min(leftDepth, rightDepth);
		return result;
	}
	int minDepth(TreeNode* root) {
		return getDepth(root);
	}
};

// class Solution {
// public:
//     int minDepth(TreeNode* root) {
//         if (root == NULL) return 0;
//         queue<TreeNode*> que;
//         que.push(root);
//         int minDepth = 0;

//         while (!que.empty()) {

//             int size = que.size();
//             minDepth++;
//             int flag = 0;

//             for (int i = 0; i < size; i++){
//                 TreeNode* node = que.front();
//                 que.pop();
//                 if (node->left) que.push(node->left);
//                 if (node->right) que.push(node->right);
//                 if (!node->left && !node->right) {
//                     flag = 1;
//                     break;
//                 }
//             }
//             if (flag == 1) break;
//         }
//         return minDepth;
//     }
// };

//559. N 叉树的最大深度
/*
// Definition for a Node.
class Node {
public:
int val;
vector<Node*> children;

Node() {}

Node(int _val) {
val = _val;
}

Node(int _val, vector<Node*> _children) {
val = _val;
children = _children;
}
};
*/

class Solution {
public:
	int maxDepth(Node* root) {
		if (root == NULL) return 0;
		queue<Node*> que;
		que.push(root);
		int Depth = 0;

		while (!que.empty()) {
			int size = que.size();
			Depth++;
			for (int i = 0; i < size; i++){
				Node* node = que.front();
				que.pop();

				int cdn_s = node->children.size();
				for (int i = 0; i < cdn_s; i++) {
					if (node->children[i]) que.push(node->children[i]);
				}
			}
		}
		return Depth;
	}
};

//104. 二叉树的最大深度
/**
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     TreeNode *left;
*     TreeNode *right;
*     TreeNode() : val(0), left(nullptr), right(nullptr) {}
*     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
*     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
* };
*/
// class Solution {
// public:
//     int maxDepth(TreeNode* root) {
//         if (root == NULL) return 0;
//         int l_Depth = maxDepth(root->left);
//         int r_Depth = maxDepth(root->right);

//         return (l_Depth > r_Depth) ? l_Depth + 1 : r_Depth + 1;
//     }
// };

// class Solution {
// public:
//     int maxDepth(TreeNode* root) {
//         if (root == nullptr) return 0;
//         queue<TreeNode*> que;
//         que.push(root);
//         int Depth = 0;

//         while (!que.empty()) {
//             int size = que.size();
//             Depth++;
//             for (int i = 0; i < size; i++) {
//                 TreeNode* node = que.front();
//                 que.pop();
//                 if (node->left) que.push(node->left);
//                 if (node->right) que.push(node->right);
//             }
//         }
//         return Depth;
//     }
// };

//101. 对称二叉树
/**
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     TreeNode *left;
*     TreeNode *right;
*     TreeNode() : val(0), left(nullptr), right(nullptr) {}
*     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
*     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
* };
*/



class Solution {
public:
	bool isSymmetric(TreeNode* root) {

		if (root == NULL)   return true;

		queue<TreeNode*> que;//栈也可以
		que.push(root->left);
		que.push(root->right);

		while (!que.empty()) {
			TreeNode *leftNode = que.front(); que.pop();
			TreeNode *rightNode = que.front(); que.pop();

			if (!leftNode && !rightNode){
				continue;
			}
			if (!leftNode || !rightNode || (leftNode->val != rightNode->val)){
				return false;
			}
			que.push(leftNode->left);
			que.push(rightNode->right);
			que.push(leftNode->right);
			que.push(rightNode->left);
		}
		return true;
	}
};

// class Solution {
// public:
//     bool compare(TreeNode* left, TreeNode* right) {
//         if (left != NULL && right == NULL)  return false;
//         else if (left == NULL && right != NULL) return false;
//         else if (left == NULL && right == NULL) return true;
//         else if (left->val != right->val)   return false;

//         bool inSide = compare(left->right, right->left);
//         bool ouSide = compare(left->left, right->right);
//         bool isSame = inSide && ouSide;

//         return isSame;
//     }
//     bool isSymmetric(TreeNode* root) {
//         if (root == NULL) return true;
//         return compare(root->left, root->right);
//     }
// };

//429. N 叉树的层序遍历
/*
// Definition for a Node.
class Node {
public:
int val;
vector<Node*> children;

Node() {}

Node(int _val) {
val = _val;
}

Node(int _val, vector<Node*> _children) {
val = _val;
children = _children;
}
};
*/

class Solution {
public:
	vector<vector<int>> levelOrder(Node* root) {
		queue<Node*> que;
		vector<vector<int>> result;

		if (root != NULL) que.push(root);

		while (!que.empty()){
			int size = que.size();
			vector<int> vec;
			for (int i = 0; i < size; i++) {
				Node *node = que.front();
				que.pop();
				vec.push_back(node->val);
				int len_children = node->children.size();
				for (int i = 0; i < len_children; i++) {
					if (node->children[i]) que.push(node->children[i]);
				}
			}
			result.push_back(vec);
		}
		return result;
	}
};

//637. 二叉树的层平均值
/*
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     TreeNode *left;
*     TreeNode *right;
*     TreeNode() : val(0), left(nullptr), right(nullptr) {}
*     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
*     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
* };
*/
class Solution {
public:
	vector<double> averageOfLevels(TreeNode* root) {
		queue<TreeNode*> que;
		vector<double> result;

		if (root != NULL) que.push(root);

		while (!que.empty()){
			int size = que.size();
			double sum = 0;
			for (int i = 0; i < size; i++){
				TreeNode* node = que.front();
				que.pop();
				sum += node->val;
				if (node->left) que.push(node->left);
				if (node->right) que.push(node->right);
			}
			result.push_back(sum / size);
		}
		return result;
	}
};

//199. 二叉树的右视图
/**
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     TreeNode *left;
*     TreeNode *right;
*     TreeNode() : val(0), left(nullptr), right(nullptr) {}
*     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
*     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
* };
*/
class Solution {
public:
	vector<int> rightSideView(TreeNode* root) {
		queue<TreeNode*> que;
		if (root != nullptr) que.push(root);
		vector<int> result;
		while (!que.empty()){
			int size = que.size();
			for (int i = 0; i < size; i++){
				TreeNode *node = que.front();
				que.pop();
				if (i == (size - 1)) result.push_back(node->val);
				if (node->left) que.push(node->left);
				if (node->right) que.push(node->right);
			}
		}
		return result;
	}
};

//107. 二叉树的层序遍历 II
/**
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     TreeNode *left;
*     TreeNode *right;
*     TreeNode() : val(0), left(nullptr), right(nullptr) {}
*     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
*     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
* };
*/
class Solution {
public:
	vector<vector<int>> levelOrderBottom(TreeNode* root) {
		queue<TreeNode*> que;
		if (root != NULL) que.push(root);
		vector<vector<int>> result;
		while (!que.empty()) {
			int size = que.size();
			vector<int>vec;
			for (int i = 0; i < size; i++) {
				TreeNode *node = que.front();
				que.pop();
				vec.push_back(node->val);
				if (node->left) que.push(node->left);
				if (node->right) que.push(node->right);
			}
			result.push_back(vec);
		}
		reverse(result.begin(), result.end());
		return result;
	}
};

//102. 二叉树的层序遍历
/**
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     TreeNode *left;
*     TreeNode *right;
*     TreeNode() : val(0), left(nullptr), right(nullptr) {}
*     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
*     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
* };
*/
class Solution {
public:
	vector<vector<int>> levelOrder(TreeNode* root) {
		queue<TreeNode*> que;
		if (root != NULL) que.push(root);
		vector<vector<int>> result;

		while (!que.empty()) {
			int size = que.size();
			vector<int> vec;

			for (int i = 0; i < size; i++) {
				TreeNode* node = que.front();
				que.pop();
				vec.push_back(node->val);
				if (node->left) que.push(node->left);
				if (node->right) que.push(node->right);
			}
			result.push_back(vec);
		}
		return result;
	}
};


//145. 二叉树的后序遍历
/**
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     TreeNode *left;
*     TreeNode *right;
*     TreeNode() : val(0), left(nullptr), right(nullptr) {}
*     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
*     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
* };
*/
class Solution {
public:
	vector<int> postorderTraversal(TreeNode* root) {
		vector<int> result;
		stack<TreeNode*> st;
		st.push(root);
		while (!st.empty()) {
			TreeNode *node = st.top();
			st.pop();

			if (node != NULL) result.push_back(node->val);
			else continue;

			st.push(node->left);
			st.push(node->right);
		}
		reverse(result.begin(), result.end());

		return result;
	}
};

//94. 二叉树的中序遍历
/**
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     TreeNode *left;
*     TreeNode *right;
*     TreeNode() : val(0), left(nullptr), right(nullptr) {}
*     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
*     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
* };
*/
class Solution {
public:
	vector<int> inorderTraversal(TreeNode* root) {
		vector<int> result;
		stack<TreeNode*> st;
		TreeNode* cur = root;

		while (cur != NULL || !st.empty()){
			if (cur != NULL){
				st.push(cur);
				cur = cur->left;
			}
			else{
				cur = st.top();
				st.pop();
				result.push_back(cur->val);
				cur = cur->right;
			}
		}
		return result;
	}
};

//144. 二叉树的前序遍历
/**
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     TreeNode *left;
*     TreeNode *right;
*     TreeNode() : val(0), left(nullptr), right(nullptr) {}
*     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
*     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
* };
*/

class Solution {
public:
	vector<int> preorderTraversal(TreeNode* root) {
		stack<TreeNode*> st;
		vector<int> result;
		st.push(root);

		while (!st.empty()) {
			TreeNode* node = st.top();
			st.pop();

			if (node != nullptr) result.push_back(node->val);
			else continue;
			st.push(node->right);
			st.push(node->left);
		}
		return result;
	}
};

// class Solution {
// public:
//     void travelsal(TreeNode* cur, vector<int>& vec){
//         if (cur == nullptr) return;

//         vec.push_back(cur->val);
//         travelsal(cur->left, vec);
//         travelsal(cur->right, vec);
//     }
//     vector<int> preorderTraversal(TreeNode* root) {
//         vector<int> result;
//         travelsal(root, result);
//         return result;
//     }
// };

//239. 滑动窗口最大值
class Solution {
private:
	class MyQueue {
	public:
		deque<int> que;
		void pop(int value){
			if (!que.empty() && value == que.front()) {
				que.pop_front();
			}
		}
		void push(int value) {
			while (!que.empty() && value > que.back()) {
				que.pop_back();
			}
			que.push_back(value);
		}

		int front() {
			return que.front();
		}
	};
public:
	vector<int> maxSlidingWindow(vector<int>& nums, int k) {
		MyQueue que;
		vector<int> result;

		for (int i = 0; i < k; i++){
			que.push(nums[i]);
		}
		result.push_back(que.front());
		int len_n = nums.size();
		for (int i = k; i < len_n; i++) {
			que.pop(nums[i - k]);
			que.push(nums[i]);
			result.push_back(que.front());
		}
		return result;
	}
};

//150. 逆波兰表达式求值
class Solution {
public:
	int evalRPN(vector<string>& tokens) {
		stack<int> st;
		int tSize = tokens.size();

		for (int i = 0; i < tSize; i++) {
			//if (tokens[i] == '+' || tokens[i] == '-' || tokens[i] == '*' || tokens[i] == '/') {错误
			if (tokens[i] == "+" || tokens[i] == "-" || tokens[i] == "*" || tokens[i] == "/") {
				int num1 = st.top();
				st.pop();
				int num2 = st.top();
				st.pop();
				if (tokens[i] == "+") st.push(num2 + num1);
				if (tokens[i] == "-") st.push(num2 - num1);//这几个判定是一个级别的不能用else if //可以else if
				if (tokens[i] == "*") st.push(num2 * num1);
				if (tokens[i] == "/") st.push(num2 / num1);
			}
			else{
				st.push(stoi(tokens[i]));
			}
		}
		int result = st.top();
		st.pop();
		return result;
	}
};

//1047. 删除字符串中的所有相邻重复项
class Solution {
public:
	string removeDuplicates(string S) {
		stack<char> st;
		for (char s : S) {
			if (st.empty() || s != st.top()){
				st.push(s);
			}
			else{
				st.pop();
			}
		}
		string result = "";
		while (!st.empty()){
			result += st.top();//+=操作会了再看
			st.pop();
		}
		reverse(result.begin(), result.end());
		return result;
	}
};

//20. 有效的括号
class Solution {
public:
	bool isValid(string s) {
		stack<int>st;
		int len_s = s.size();
		for (int i = 0; i < len_s; i++) {
			if (s[i] == '(') st.push(')');
			else if (s[i] == '[') st.push(']');
			else if (s[i] == '{') st.push('}');
			else if (st.empty() || st.top() != s[i]) return false;
			else st.pop();
		}
		return st.empty();
	}
};

//225. 用队列实现栈
class MyStack {
public:
	queue<int>que1;
	queue<int>que2;
	/** Initialize your data structure here. */
	MyStack() {

	}

	/** Push element x onto stack. */
	void push(int x) {
		que1.push(x);
	}

	/** Removes the element on top of the stack and returns that element. */
	int pop() {
		int size = que1.size();
		size--;
		while (size--) {
			que2.push(que1.front());
			que1.pop();
		}
		int result = que1.front();
		que1.pop();
		que1 = que2;

		while (!que2.empty()) {
			que2.pop();
		}
		return result;
	}

	/** Get the top element. */
	int top() {
		return que1.back();
	}

	/** Returns whether the stack is empty. */
	bool empty() {
		return que1.empty();
	}
};

/**
* Your MyStack object will be instantiated and called as such:
* MyStack* obj = new MyStack();
* obj->push(x);
* int param_2 = obj->pop();
* int param_3 = obj->top();
* bool param_4 = obj->empty();
*/

//232. 用栈实现队列
class MyQueue {
public:
	/** Initialize your data structure here. */
	stack<int> stIn;
	stack<int> stOut;
	MyQueue() {

	}

	/** Push element x to the back of queue. */
	void push(int x) {
		stIn.push(x);
	}

	/** Removes the element from in front of queue and returns that element. */
	int pop() {
		if (stOut.empty()){
			while (!stIn.empty()){
				stOut.push(stIn.top());
				stIn.pop();
			}
		}
		int result = stOut.top();
		stOut.pop();
		return result;
	}

	/** Get the front element. */
	int peek() {
		int result = this->pop();
		stOut.push(result);
		return result;
	}

	/** Returns whether the queue is empty. */
	bool empty() {
		return stIn.empty() && stOut.empty();
	}
};

/**
* Your MyQueue object will be instantiated and called as such:
* MyQueue* obj = new MyQueue();
* obj->push(x);
* int param_2 = obj->pop();
* int param_3 = obj->peek();
* bool param_4 = obj->empty();
*/

//2021_3_21
//73. 矩阵置零
class Solution {
public:
	void setZeroes(vector<vector<int>>& matrix) {
		int rowSize = matrix.size();
		int colSize = matrix[0].size();
		vector<int>row(rowSize), col(colSize);
		for (int i = 0; i < rowSize; i++){
			for (int j = 0; j < colSize; j++){
				if (!matrix[i][j]){
					row[i] = col[j] = true;
				}
			}
		}
		for (int i = 0; i < rowSize; i++){
			for (int j = 0; j < colSize; j++){
				if (row[i] || col[j])
					matrix[i][j] = 0;
			}
		}
	}
};

//2021_3_20
//28. 实现 strStr()
class Solution {
public:

	void GetNext(int *next, const string & needle){
		int j = -1;
		next[0] = j;
		int len = needle.size();
		for (int i = 1; i < len; i++){

			while (j >= 0 && needle[j + 1] != needle[i])
				j = next[j];

			if (needle[i] == needle[j + 1])
				j++;

			next[i] = j;
		}
	}

	int strStr(string haystack, string needle) {
		if (needle.size() == 0)
			return 0;
		int next[needle.size()];
		GetNext(next, needle);
		int j = -1;
		int hay_len = haystack.size();
		for (int i = 0; i < hay_len; i++){

			while (j >= 0 && haystack[i] != needle[j + 1]){
				j = next[j];
			}
			if (haystack[i] == needle[j + 1])
				j++;
			if (j == (needle.size() - 1)){
				return (i - needle.size() + 1);
			}
		}
		return -1;
	}
};

//2021_3_19
//242. 有效的字母异位词
class Solution {
public:
	bool isAnagram(string s, string t) {
		int ret[26]{};
		for (int i = 0; i < s.size(); ++i)
			ret[s[i] - 'a']++;

		for (int i = 0; i < t.size(); ++i)
			ret[t[i] - 'a']--;

		for (int i = 0; i < 26; i++){
			if (ret[i] != 0)
				return false;
		}
		return true;
	}
};

//142. 环形链表 II
/**
* Definition for singly-linked list.
* struct ListNode {
*     int val;
*     ListNode *next;
*     ListNode(int x) : val(x), next(NULL) {}
* };
*/
class Solution {
public:
	ListNode *detectCycle(ListNode *head) {
		// if (head == nullptr || head->next = nullptr)
		//     return -1;
		ListNode *fast = head;
		//ListNode *fast = head->next;根据公式刚好差一个永远无法相遇
		ListNode *slow = head;

		while (fast && fast->next){
			fast = fast->next->next;
			slow = slow->next;
			//if (fast->val == slow->val){//值是有可能重复的
			if (fast == slow){
				ListNode *index1 = head;
				ListNode *index2 = fast;
				while (index1 != index2){
					index1 = index1->next;
					index2 = index2->next;
				}
				return index2;
			}
		}
		return NULL;
	}
};

//206. 反转链表
/**
* Definition for singly-linked list.
* struct ListNode {
*     int val;
*     ListNode *next;
*     ListNode() : val(0), next(nullptr) {}
*     ListNode(int x) : val(x), next(nullptr) {}
*     ListNode(int x, ListNode *next) : val(x), next(next) {}
* };
*/
// class Solution {
// public:
//     ListNode* reverseList(ListNode* head) {
//         ListNode *temp;
//         ListNode *cur = head;
//         ListNode *pre = NULL;
//         while (cur){
//             temp = cur->next;
//             cur->next = pre;
//             pre = cur;
//             cur = temp;//
//         }
//         return pre;
//     }
// };

class Solution {
public:
	ListNode* reverse(ListNode *pre, ListNode *cur){
		if (cur == NULL)
			return pre;
		ListNode* temp = cur->next;
		cur->next = pre;

		return reverse(cur, temp);
	}

	ListNode* reverseList(ListNode* head) {
		ListNode* cur = head;
		ListNode* pre = NULL;

		return reverse(pre, cur);
	}
};

//203. 移除链表元素
/**
* Definition for singly-linked list.
* struct ListNode {
*     int val;
*     ListNode *next;
*     ListNode() : val(0), next(nullptr) {}
*     ListNode(int x) : val(x), next(nullptr) {}
*     ListNode(int x, ListNode *next) : val(x), next(next) {}
* };
*/

class Solution {
public:
	ListNode* removeElements(ListNode* head, int val) {

		while (head != NULL && head->val == val) { // 注意这里不是if
			ListNode* tmp = head;
			head = head->next;
			delete tmp;
		}

		ListNode* cur = head;
		while (cur != NULL && cur->next != NULL) {//[7,7,7,7] 7
			if (cur->next->val == val) {
				ListNode* tmp = cur->next;
				cur->next = cur->next->next;
				delete tmp;
			}
			else {
				cur = cur->next;
			}
		}
		return head;
	}
};

//  class Solution {
// public:
//     ListNode* removeElements(ListNode* head, int val) {
//         ListNode *dummy = new ListNode(0);
//         dummy->next = head;
//         ListNode *cur = dummy;

//         while (cur->next != nullptr){

//             if (cur->next->val == val){

//                 ListNode *tmp = cur->next;
//                 cur->next = cur->next->next;
//                 delete tmp;
//             }
//             else
//                 cur = cur->next;
//         }
//         return dummy->next;
//     }
// };

// class Solution {
// public:
//     ListNode* removeElements(ListNode* head, int val) {

//         ListNode *dummy = new ListNode(0);
//         dummy->next = head;

//         ListNode *pre = dummy;
//         ListNode *cur = head;

//         while (cur != nullptr){

//             if (cur->val == val){
//                 pre->next = cur->next;
//                 cur = cur->next;
//             }
//             else{
//             cur = cur->next;
//             pre = pre->next;
//             }
//         }
//         return dummy->next;
//     }
// };

//剑指 Offer 05. 替换空格
class Solution {
public:
	string replaceSpace(string s) {
		int oldSize = s.size();
		int count = 0;

		for (int i = 0; i < oldSize; i++){

			if (s[i] == ' ')
				count++;
		}

		//resize(s.size(), count * 2);//错误
		s.resize(s.size() + count * 2);
		int newSize = s.size();

		//for (int i = newSize - 1, j = oldSize - 1; j >= 0; --i, --j){//脑子想不到下面的
		for (int i = newSize - 1, j = oldSize - 1; j < i; --i, --j){

			if (s[j] != ' ')
				s[i] = s[j];

			else{
				s[i] = '0';
				s[i - 1] = '2';
				s[i - 2] = '%';
				i -= 2;
			}
		}
		return s;
	}
};

//541. 反转字符串 II
// class Solution {
// public:
//     string reverseStr(string s, int k) {

//         for (int i = 0 ; i < s.size(); i+= (2 * k)){

//             if (i + k <= s.size()){//等号的问题，设置一个k = size 的情况看

//                 reverse(s.begin() + i, s.begin() + i + k);
//                 continue;
//             }

//             reverse(s.begin() + i, s.begin() + s.size());
//         }
//         return s;
//     }
// };//reverse 函数的用法

class Solution {
public:

	void reverse(string &s, int start, int end){
		int offest = (end - start + 1) / 2; //start的下标不是固定为0

		//for (start; start < (start + offest); start++, end--)//这样写是、start + offest 的判定条件永远不会终止
		for (int i = start, j = end; i < (start + offest); ++i, --j)
			swap(s[i], s[j]);
	}

	string reverseStr(string s, int k) {

		for (int i = 0; i < s.size(); i += (2 * k)){

			if (i + k <= s.size()){

				reverse(s, i, i + k - 1);
				continue;
			}

			reverse(s, i, s.size() - 1);
		}
		return s;
	}
};

//344. 反转字符串
// class Solution {
// public:
//     void reverseString(vector<char>& s) {
//         int left = 0, right = s.size() - 1;

//         while (left < right){

//             int temp = s[left];
//             s[left] = s[right];
//             s[right] = temp;

//             left++, right--;
//         }
//     }
// };

class Solution {
public:
	void reverseString(vector<char>& s) {
		for (int i = 0, j = s.size() - 1; i < s.size() / 2; i++, j--)
			swap(s[i], s[j]);
	}
};

//59. 螺旋矩阵 II
class Solution {
public:
	vector<vector<int>> generateMatrix(int n) {
		vector<vector<int>> res(n, vector<int>(n, 0));
		int startx = 0;
		int starty = 0;
		int count = 1;
		int loop = n / 2;
		int mid = n / 2;
		int offest = 1;
		int i, j;
		while (loop--){

			for (j = starty; j < starty + n - offest; j++)
				res[startx][j] = count++;

			for (i = startx; i < startx + n - offest; i++)
				res[i][j] = count++;

			for (; j > starty; j--)
				res[i][j] = count++;

			for (; i > startx; i--)
				res[i][j] = count++;

			startx++;
			starty++;

			offest += 2;
		}

		if (n % 2)
			res[mid][mid] = count;
		return res;

	}
};

//1603. 设计停车系统
class ParkingSystem {
public:

	int _big, _medium, _small;
	ParkingSystem(int big, int medium, int small) {
		_big = big;
		_medium = medium;
		_small = small;
	}
	bool addCar(int carType) {

		if (carType == 1){
			if (_big > 0){
				_big--;
				return true;
			}
		}
		else if (carType == 2){
			if (_medium > 0){
				_medium--;
				return true;
			}
		}
		else if (carType == 3){
			if (_small > 0){
				_small--;
				return true;
			}
		}
		return false;
	}
};

/**
* Your ParkingSystem object will be instantiated and called as such:
* ParkingSystem* obj = new ParkingSystem(big, medium, small);
* bool param_1 = obj->addCar(carType);
*/

//209. 长度最小的子数组
// class Solution {
// public:
//     int minSubArrayLen(int target, vector<int>& nums) {
//         int result = INT32_MAX;//为了第一次 len 的赋值
//         int len = 0;
//         int sum = 0;
//         int size = nums.size();

//         for (int i = 0; i < size; i++){
//             sum = 0;
//             for (int j = i; j < size; j++){
//                 //sum += nums[i];
//                 sum += nums[j];
//                 if (sum >= target){
//                     len = j - i + 1;
//                     result = result < len ? result : len;
//                     break;//这次循环可以结束了
//                 }
//             }
//         }
//         return result == INT32_MAX ? 0 : result;
//     }
// };

class Solution {
public:
	int minSubArrayLen(int target, vector<int>& nums) {
		int result = INT32_MAX;
		int size = nums.size();
		int len = 0;
		int sum = 0;
		int i = 0;

		for (int j = 0; j < size; j++){
			sum += nums[j];

			while (sum >= target){
				len = (j - i + 1);
				result = result < len ? result : len;
				sum -= nums[i++];
			}
		}
		return result == INT32_MAX ? 0 : result;
	}
};

//27. 移除元素
// class Solution {
// public:
//     int removeElement(vector<int>& nums, int val) {
//         int size = nums.size();
//         for (int i = 0; i < size; i++){

//             if (nums[i] == val){

//                 for (int j = i; j < size - 1; j++)
//                     nums[j] = nums[j + 1];
//                 // for (int j = i + 1; j < size; j++)
//                 //     nums[j - 1] = nums[j];
//                 i--;
//                 size--;
//             }
//         }
//         return size;
//     }
// };

class Solution {
public:
	int removeElement(vector<int>& nums, int val) {
		int fast = 0, slow = 0;
		int size = nums.size();

		while (fast < size){

			if (nums[fast] == val)
				++fast;
			else
				nums[slow++] = nums[fast++];
		}
		return slow;
	}
};


// class Solution {
// public:
//     int removeElement(vector<int>& nums, int val) {
//             int n = nums.size();
//             int count = 0;
//             for (int i = 0; i < n; i++){
//                 if (nums[i] != val)
//                     nums[count++] = nums[i];
//             }
//             return count;
//     }
// };

//35. 搜索插入位置
// class Solution {
// public:
//     int searchInsert(vector<int>& nums, int target) {
//         int left = 0, right = nums.size() - 1;
//         while (left <= right){
//             int mid = left + right;
//             if (nums[mid] < target)
//                 left = mid + 1;
//             else if (nums[mid] > target)
//                 right = mid - 1;
//             else return mid;
//         }
//         return left;//return right + 1
//     }
// };

// class Solution {
// public:
//     int searchInsert(vector<int>& nums, int target) {
//         int n = nums.size();
//         int left = 0;
//         int right = n;
//         while (left < right){
//             int mid = left + (right - left) / 2;
//             if (nums[mid] > target)
//                 right = mid;
//             else if (nums[mid] < target)
//                 left = mid + 1;
//             else return mid;
//         }
//         return right;
//     }
// };

// class Solution {
// public:
//     int searchInsert(vector<int>& nums, int target) {
//         int n = nums.size();
//         int left = 0;
//         int right = n - 1;//[1,3,5,6],7 这种情况会出错
//         while (left < right){
//             int mid = left + (right - left) / 2;
//             if (nums[mid] > target)
//                 right = mid;
//             else if (nums[mid] < target)
//                 left = mid + 1;
//             else return mid;
//         }
//         return right;
//     }
// };

// class Solution {
// public:
//     int searchInsert(vector<int>& nums, int target) {
//         for (int i = 0; i < nums.size(); i++){
//             if (nums[i] >= target)
//                 return i;
//         }
//         return nums.size();
//     }
// };

//92. 反转链表 II
/**
* Definition for singly-linked list.
* struct ListNode {
*     int val;
*     ListNode *next;
*     ListNode() : val(0), next(nullptr) {}
*     ListNode(int x) : val(x), next(nullptr) {}
*     ListNode(int x, ListNode *next) : val(x), next(next) {}
* };
*/
class Solution {
private:
	void reverse(ListNode *head){
		ListNode *pre = nullptr;
		ListNode *cur = head;
		while (cur != nullptr){
			ListNode *next = cur->next;
			cur->next = pre;
			pre = cur;
			cur = next;
		}

	}
public:
	ListNode* reverseBetween(ListNode* head, int left, int right) {
		listNode *dummyNode = new ListNode(-1);
		dummyNode->next = head;

		ListNode *pre = dummyNode;
		for (int i = 0; i < left - 1; i++)
			pre = pre->next;

	}
};

//2021_3_15
//925. 长按键入
class Solution {
public:
	bool isLongPressedName(string name, string typed) {
		int i = 0, j = 0;
		while (j < typed.length()){
			if (i < name.length() && name[i] == typed[j])
				i++, j++;
			else if (j > 0 && typed[j] == typed[j - 1])
				j++;
			else return false;
		}
		return i == name.length();
	}
};

//2021_3_14
//141. 环形链表
// class Solution {
// public:
//     bool hasCycle(ListNode *head) {
//         if(head == nullptr || head->next == nullptr)
//             return false;
//             ListNode *fast = head->next;
//             ListNode *slow = head;
//         while(slow != fast){
//             if(fast == nullptr || fast->next == nullptr)
//                 return false;
//             fast = fast->next->next;
//             slow = slow->next;
//         }
//         return true;
//     }
// };

class Solution{
public:
	bool hasCycle(ListNode *head) {
		if (head == nullptr || head->next == nullptr)
			return false;
		ListNode *fast = head;
		ListNode *slow = head;
		while (fast && fast->next){
			fast = fast->next->next;
			slow = slow->next;
			if (fast == slow)
				return true;
		}
		return false;
	}
};

//88. 合并两个有序数组
class Solution {
public:
	void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
		for (int i = 0; i != n; i++){
			nums1[m + i] = nums2[i];
		}
		sort(nums1.begin(), nums1.end());
	}
};

class Solution {
public:
	void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
		int p1 = 0, p2 = 0;
		//int count = 0;
		int tmp[m + n];
		int cur;

		while (p1 <m || p2<n){
			if (p1 == m)
				cur = nums2[p2++];
			else if (p2 == n)
				cur = nums1[p1++];
			else if (nums1[p1] < nums2[p2])
				cur = nums1[p1++];
			else
				cur = nums2[p2++];
			//tmp[count++] = cur;
			tmp[p1 + p2 - 1] = cur;
		}
		for (int i = 0; i < m + n; i++)
			nums1[i] = tmp[i];
	}
};

class Solution {
public:
	void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
		int p1 = m - 1, p2 = n - 1;
		int tail = m + n - 1;
		int cur;
		while (p1 >= 0 || p2 >= 0){
			if (p1 < 0)
				cur = nums2[p2--];
			else if (p2 < 0)
				cur = nums1[p1--];
			else if (nums1[p1] < nums2[p2])
				cur = nums2[p2--];
			else cur = nums1[p1--];
			nums1[tail--] = cur;
		}
	}
};

//344.反转字符串
class Solution {
public:
	void reverseString(vector<char>& s) {
		int n = s.size();
		int left = 0;
		int right = n - 1;
		int tmp;
		while (left < right){
			tmp = s[left];
			s[left] = s[right];
			s[right] = tmp;
			left++, right--;
		}
	}
};

class Solution {
public:
	void reverseString(vector<char>& s) {
		int n = s.size();
		int left = 0;
		int right = n - 1;
		while (left < right){
			swap(s[left], s[right]);
			left++, right--;
		}
	}
};

class Solution {
public:
	void reverseString(vector<char>& s) {
		int n = s.size();
		char tmp;
		for (int i = 0; i < n / 2; i++){
			tmp = s[i];
			s[i] = s[n - 1 - i];
			s[n - 1 - i] = tmp;
		}
	}
};

//26. 删除排序数组中的重复项
class Solution {
public:
	int removeDuplicates(vector<int>& nums) {
		if (nums.empty())
			return 0;
		int i = 0;
		for (int j = 1; j < nums.size(); ++j){
			if (nums.at(i) != nums.at(j))
				nums.at(++i) = nums.at(j);
		}
		return i + 1;
	}
};

int main()
{
	return 0;
}