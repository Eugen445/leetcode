#include<iostream>
using namespace std;

//2021_4_5

//279. ��ȫƽ����
class Solution {
public:
	int numSquares(int n) {
		vector<int> dp(n + 1, INT_MAX);
		dp[0] = 0;
		dp[1] = 1;
		//for (int i = 1; i < n/2; ++i) {//������1
		for (int i = 1; i <= n / 2; ++i) {//����ط�д<=n�Ͳ���Ҫ��n�����⴦��
			int tmp = i * i;
			for (int j = tmp; j <= n; ++j) {
				if (dp[j - tmp] != INT_MAX) {
					dp[j] = min(dp[j], dp[j - tmp] + 1);
				}
			}
		}
		if (dp[n] == INT_MAX) return -1;
		return dp[n];
	}
};

//322. ��Ǯ�һ�
class Solution {
public:
	int coinChange(vector<int>& coins, int amount) {
		vector<int> dp(amount + 1, INT_MAX);
		dp[0] = 0;
		for (int i = 0; i < coins.size(); ++i) { //�ȱ�����Ʒ
			for (int j = coins[i]; j <= amount; ++j) {
				if (dp[j - coins[i]] != INT_MAX) {
					dp[j] = min(dp[j], dp[j - coins[i]] + 1);
				}
			}
		}
		if (dp[amount] == INT_MAX) return -1;
		return dp[amount];
	}
};

//2021_4_4

//70. ��¥��
class Solution {
public:
	int climbStairs(int n) { //��ȫ��������
		vector<int> dp(n + 1, 0);
		dp[0] = 1;
		for (int i = 1; i <= n; ++i) {
			for (int j = 1; j <= 2; ++j) { //2���Ի���m
				if (i - j >= 0) dp[i] += dp[i - j];
			}
		}
		return dp[n];
	}
};

//377. ����ܺ� ��
class Solution {
public:
	int combinationSum4(vector<int>& nums, int target) {
		vector<int> dp(target + 1, 0);
		dp[0] = 1;
		for (int i = 0; i <= target; ++i) {
			for (int j = 0; j < nums.size(); ++j) {
				if (i - nums[j] >= 0 && dp[i] < INT_MAX - dp[i - nums[j]]) { //����dp�Ŀ����Դ���int��Χ���������
					dp[i] += dp[i - nums[j]];
				}
			}
		}
		return dp[target];
	}
};

//518. ��Ǯ�һ� II
class Solution {
public:
	int change(int amount, vector<int>& coins) {
		vector<int> dp(amount + 1, 0);
		dp[0] = 1;
		for (int i = 0; i < coins.size(); ++i) {
			for (int j = coins[i]; j <= amount; ++j) {
				dp[j] += dp[j - coins[i]];
			}
		}
		return dp[amount];
	}
};

//781. ɭ���е�����
class Solution {
public:
	int numRabbits(vector<int>& answers) {
		if (answers.size() == 0) return 0;
		vector<int> dp(answers.size(), 0);
		sort(answers.begin(), answers.end());
		dp[0] = answers[0] + 1;
		//��һ�ο��ǣ�˼���µ�һ���ύ�ĳ�ʼ���Ƿ��ܹ������ݴ�

		int count = 0; //������
		for (int i = 1; i < answers.size(); ++i) {
			if (answers[i] > answers[i - 1]) {
				dp[i] = dp[i - 1] + answers[i] + 1;
				count = 0;
			}
			else{
				//
				count++;
				if (count <= answers[i]) {
					//count++;//λ��д����
					dp[i] = dp[i - 1];
				}
				else {
					count = 0;
					dp[i] = dp[i - 1] + answers[i] + 1;
				}
			}
		}
		return dp[answers.size() - 1];
	}
};

//2021_4_3

//��ָ Offer 64. ��1+2+��+n
class Solution {
public:
	int sumNums(int n) {
		bool aha[n][n + 1];
		return sizeof(aha) >> 1;
	}
};

//������ 17.12. BiNode
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
private:
	TreeNode* dummy = new TreeNode(0);
	TreeNode* cur = dummy;
	void sort(TreeNode* root) {
		if (root == nullptr) return;
		sort(root->left);
		root->left = nullptr;
		cur->right = root;
		cur = root;
		sort(root->right);
	}
public:
	TreeNode* convertBiNode(TreeNode* root) {
		sort(root);
		return dummy->right;
	}
};

//������ 08.01. ��������
class Solution {
public:
	int waysToStep(int n) {
		if (n < 3) return n;
		vector<size_t> dp(n + 1, 0);
		dp[0] = 1;
		dp[1] = 1;
		dp[2] = 2;
		for (int i = 3; i <= n; ++i) {
			dp[i] = dp[i - 1] % 1000000007 + dp[i - 2] % 1000000007 + dp[i - 3] % 1000000007;
		}
		return dp[n] % 1000000007;
	}
};

//������ 01.02. �ж��Ƿ�Ϊ�ַ�����
class Solution {
public:
	bool CheckPermutation(string s1, string s2) {
		sort(s1.begin(), s1.end());
		sort(s2.begin(), s2.end());
		for (int i = 0; i < s2.size(); ++i) {
			if (s1[i] != s2[i]) return false;
		}
		return true;
	}
};

//������ 17.04. ��ʧ������
class Solution {
public:
	int missingNumber(vector<int>& nums) {
		int res = 0;
		for (int i = 1; i <= nums.size(); ++i) {
			res ^= nums[i - 1];
			res ^= i;
		}
		return res;
	}
};

//494. Ŀ���
class Solution {
public:
	int findTargetSumWays(vector<int>& nums, int S) {
		int sum = 0;
		for (int i = 0; i < nums.size(); ++i) sum += nums[i];
		if (sum < S) return 0;
		//if ((sum + S) / 2 == 1) return 0;
		if ((sum + S) % 2 == 1) return 0;
		int bag_size = (sum + S) / 2;
		vector<int> dp(bag_size + 1, 0);
		dp[0] = 1;
		for (int i = 0; i < nums.size(); ++i) {
			//for (int j = bag_size; j>=0; j--) {
			for (int j = bag_size; j >= nums[i]; j--) {
				dp[j] += dp[j - nums[i]];
			}
		}
		return dp[bag_size];
	}
};

//1049. ���һ��ʯͷ������ II
class Solution {
public:
	int lastStoneWeightII(vector<int>& stones) {
		vector<int> dp(15001, 0);
		int sum = 0;
		for (int i = 0; i < stones.size(); ++i) sum += stones[i];
		int target = sum / 2;
		for (int i = 0; i < stones.size(); ++i) {
			for (int j = target; j >= stones[i]; --j) {
				dp[j] = max(dp[j], dp[j - stones[i]] + stones[i]);
			}
		}
		return sum - dp[target] - dp[target];
	}
};

//1143. �����������
class Solution {
public:
	int longestCommonSubsequence(string text1, string text2) {
		vector<vector<int>> dp(text1.size() + 1, vector<int>(text2.size() + 1, 0));
		int text1_size = text1.size();
		int text2_size = text2.size();
		for (int i = 1; i <= text1_size; ++i) { //=����
			for (int j = 1; j <= text2_size; ++j) {
				if (text1[i - 1] == text2[j - 1]) {
					dp[i][j] = dp[i - 1][j - 1] + 1;
				}
				else {
					dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
				}
			}
		}
		return dp[text1_size][text2_size];
	}
};

//2021_4_2

//��ָ Offer 27. �������ľ���
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
private:
	void mirror_tree(TreeNode* root) {
		if (root == nullptr) return;
		swap(root->left, root->right);
		mirrorTree(root->left);
		mirrorTree(root->right);
	}
public:
	TreeNode* mirrorTree(TreeNode* root) {
		mirror_tree(root);
		return root;
	}
};

//2021_4_1

//416. �ָ�Ⱥ��Ӽ�
class Solution {
public:
	bool canPartition(vector<int>& nums) {
		int sum = 0;
		vector<int> dp(10001, 0);
		int size_nums = nums.size();
		for (int i = 0; i < size_nums; ++i)
			sum += nums[i];
		//if (sum/2 == 1) return false;
		if (sum % 2 == 1) return false;
		int target = sum / 2;
		for (int i = 0; i < size_nums; ++i)
		for (int j = target; j >= nums[i]; --j)
			dp[j] = max(dp[j], dp[j - nums[i]] + nums[i]);
		if (dp[target] == target) return true;
		return false;
	}
};

//������ 17.21. ֱ��ͼ��ˮ��
class Solution {
public:
	int trap(vector<int>& height) {
		if (height.size() <= 2) return 0;
		vector<int> max_left(height.size(), 0);
		vector<int> max_right(height.size(), 0);
		int size = height.size();

		max_left[0] = height[0];
		for (int i = 1; i < size; ++i)
			max_left[i] = max(height[i], max_left[i - 1]);

		max_right[size - 1] = height[size - 1];
		for (int i = size - 2; i >= 0; --i)
			max_right[i] = max(height[i], max_right[i + 1]);

		int sum = 0;
		for (int i = 0; i < size; ++i) {
			int count = min(max_left[i], max_right[i]) - height[i];
			if (count > 0) sum += count;
		}
		return sum;
	}
};

class Solution {
public:
	int trap(vector<int>& height) {
		int sum = 0;
		int len = height.size();
		for (int i = 0; i < len; ++i) {
			if (i == 0 || i == len - 1) continue;
			int r_height = height[i];
			int l_height = height[i];
			for (int r = i + 1; r < len; ++r)
			if (height[r] > r_height) r_height = height[r];
			for (int l = i - 1; l >= 0; --l)
			if (height[l] > l_height) l_height = height[l];
			int h = min(l_height, r_height) - height[i];
			if (h > 0) sum += h;
		}
		return sum;
	}
};


//������ 08.03. ħ������
class Solution {
public:
	int findMagicIndex(vector<int>& nums) {
		int left = 0;
		int right = nums.size() - 1;
		sort(nums.begin(), nums.end());
		while (left <= right) {
			int mid = left + (right - left) / 2;
			if (nums[mid] > mid) right = mid - 1;
			else if (nums[mid] < mid) left = mid + 1;
			return mid;
		}
		return -1;
	}
};

//������ 02.07. �����ཻ
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
	ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {

		if (!headA || !headB) return nullptr; //�����ж���������

		ListNode* curA_B = headA;
		ListNode* curB_A = headB;

		while (curA_B || curB_A) {

			if (curA_B == nullptr) curA_B = headB;
			if (curB_A == nullptr) curB_A = headA;
			if (curA_B == curB_A) return curA_B;
			curA_B = curA_B->next;
			curB_A = curB_A->next;
		}
		return nullptr;
	}
};

//������ 05.07. ��Խ���
class Solution {
public:
	int exchangeBits(int num) {
		int even = (num & 0xaaaaaaaa) >> 1;
		int odd = (num & 0x55555555) << 1;
		return even | odd;
	}
};

//������ 03.04. ��ջΪ��
class MyQueue {
private:
	stack<int> st1;
	stack<int> st2;
public:
	/** Initialize your data structure here. */
	MyQueue() {

	}

	/** Push element x to the back of queue. */
	void push(int x) {
		st1.push(x);
	}

	/** Removes the element from in front of queue and returns that element. */
	int pop() {
		if (st2.empty()) {
			while (!st1.empty()) { st2.push(st1.top()); st1.pop(); }
		}
		int res = st2.top(); st2.pop();
		return res;
	}

	/** Get the front element. */
	int peek() {
		if (st2.empty()) {
			while (!st1.empty()) { st2.push(st1.top()); st1.pop(); }
		}
		return st2.top();
	}

	/** Returns whether the queue is empty. */
	bool empty() {
		return st1.empty() && st2.empty();
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


//������ 01.01. �ж��ַ��Ƿ�Ψһ
class Solution {
public:
	bool isUnique(string astr) {
		sort(astr.begin(), astr.end());
		int len = astr.size();
		for (int i = 0; i < len - 1; ++i) {
			if (astr[i] == astr[i + 1]) return false;
		}
		return true;
	}
};

class Solution {
public:
	bool isUnique(string astr) {
		int len = astr.size();
		for (int i = 0; i < len; ++i) {
			for (int j = i + 1; j < len; ++j) {
				if (astr[i] == astr[j]) return false;
			}
		}
		return true;
	}
};


//������ 02.02. ���ص����� k ���ڵ�
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
	int kthToLast(ListNode* head, int k) {

		ListNode* p = head;
		ListNode* q = head;
		while (k--) p = p->next;
		while (p) p = p->next, q = q->next;

		return q->val;
	}
};

//������ 04.02. ��С�߶���
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
private:
	TreeNode* S_A_BST(vector<int>& nums, int left, int right) {
		if (left >= right) return nullptr;
		int mid = left + (right - left) / 2;
		TreeNode* root = new TreeNode(nums[mid]);
		root->left = S_A_BST(nums, left, mid);
		root->right = S_A_BST(nums, mid + 1, right);

		return root;
	}
public:
	TreeNode* sortedArrayToBST(vector<int>& nums) {
		return S_A_BST(nums, 0, nums.size()); //����ҿ�����
	}
};

//������ 02.03. ɾ���м�ڵ�
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
	void deleteNode(ListNode* node) {
		node->val = node->next->val;
		node->next = node->next->next;
	}
};

//654. ��������
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
private:
	TreeNode* con_max_BT(vector<int>& nums, int left, int right) {
		if (left == right) return nullptr;
		//if (left >= right) return nullptr;//������//������յ�

		int maxindex = left;
		for (int i = left + 1; i < right; ++i) {
			if (nums[i] > nums[maxindex]) maxindex = i;
		}
		TreeNode* root = new TreeNode(nums[maxindex]);
		root->left = con_max_BT(nums, left, maxindex);
		root->right = con_max_BT(nums, maxindex + 1, right);

		return root;
	}
public:
	TreeNode* constructMaximumBinaryTree(vector<int>& nums) {
		return con_max_BT(nums, 0, nums.size());
	}
};

//������ 04.03. �ض���Ƚڵ�����
/**
* Definition for a binary tree node.
* struct TreeNode {
*     int val;
*     TreeNode *left;
*     TreeNode *right;
*     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
* };
*/
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
	vector<ListNode*> listOfDepth(TreeNode* tree) {
		vector<ListNode*> res;
		queue<TreeNode*> que;
		if (tree != nullptr) que.push(tree);

		ListNode dummyNode(0);
		while (!que.empty()) {

			int size = que.size();
			ListNode *tmp = &dummyNode;

			for (int i = 0; i < size; ++i) {

				TreeNode *cur = que.front(); que.pop();
				if (cur->left) que.push(cur->left);
				if (cur->right) que.push(cur->right);
				tmp->next = new ListNode(cur->val);
				tmp = tmp->next;
			}
			res.push_back(dummyNode.next);
		}
		return res;
	}
};

//��ָ Offer 26. �����ӽṹ
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
private:
	bool is_same(TreeNode* A, TreeNode* B) {

		if (B == nullptr) return true; //B��ƥ������ 
		if (A == nullptr) return false; //A��ƥ���꣬��Bȴ���нڵ�
		if (A->val != B->val) return false;

		return  is_same(A->left, B->left) && is_same(A->right, B->right);
	}
public:
	bool isSubStructure(TreeNode* A, TreeNode* B) {

		if (A == nullptr || B == nullptr) return false;
		bool res = false;
		if (A->val == B->val) res = is_same(A, B); //ֵ��ȲŻ�ȥ�ж��Ƿ�Ϊ�ӽṹ
		if (!res) res = isSubStructure(A->left, B);
		if (!res) res = isSubStructure(A->right, B);

		return res;
	}
};

//1006. ���׳�
class Solution {
public:
	int clumsy(int N) {
		if (N == 1) {
			return 1;
		}
		else if (N == 2) {
			return 2;
		}
		else if (N == 3) {
			return 6;
		}
		else if (N == 4) {
			return 7;
		}

		if (N % 4 == 0) {
			return N + 1;
		}
		else if (N % 4 <= 2) {
			return N + 2;
		}
		else {
			return N - 1;
		}
	}
};

//343. �������
class Solution {
public:
	int integerBreak(int n) { //��̬�滮
		vector<int> dp(n + 1);
		dp[2] = 1;
		for (int i = 3; i <= n; ++i) {
			for (int j = 1; j < i - 1; ++j) {//ΪʲôҪС��i - 1
				//j < i - 1 ����Ϊȡ�� j = i - 1�� 1������������j = 1��i = i - 1���� //������дһ��//д��Ҳ��Ӱ����
				dp[i] = max(dp[i], max((i - j) * j, dp[i - j] * j)); //Ϊʲô����д��
				//dp[i - j]��ʵ�����Ѿ���ֺõ�i - j �����ֵ��
				//��Ϊ�������� i - j �ı�С����ʵ����չʾ i - j ֵ�����Ž�Ĺ��̣�ֻ���j�Ϳ����˲���Ҫ�ڷֽ�j��
				//�ڶ����Ҳ�̫���//��ʱ������Ϊ����i��ֵС��ʱ�����ҳ����зָ�����Ա�i�����ʹ��
				//�о�����ȫ�ǵ������ڷֽ���
				//��dp[7]��ʱ��ڶ����͵����������ֵ�Ѿ���ƽ
				//��dp[8]�ǵ�����ʽ��վ��Ҫ��λ��
				//��dp[9]��ʱ����ȫ��������
				//��dp[10]��ʱ����ȫ��������
				//�ܽ�ڶ���ʽ��ֻ��7��7֮ǰ�����ã����濿������ʽ��
			}
		}
		return dp[n];
	}
};

//63. ��ͬ·�� II
class Solution {
public:
	int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
		int m = obstacleGrid.size();
		int n = obstacleGrid[0].size();
		vector<vector<int>> dp(m, vector<int>(n, 0));
		for (int i = 0; i < m && obstacleGrid[i][0] == 0; ++i) dp[i][0] = 1; //�����ϰ�ֱ�Ӳ��ü���
		for (int j = 0; j < n && obstacleGrid[0][j] == 0; ++j) dp[0][j] = 1; //�����ϰ�ֱ�Ӳ��ü���
		for (int i = 1; i < m; ++i) {
			for (int j = 1; j < n; ++j) {
				if (obstacleGrid[i][j] == 1) continue; //������ζ�����λ�õķ�����0��
				dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
			}
		}
		return dp[m - 1][n - 1];
	}
};

//62. ��ͬ·��
// class Solution {
// public:
//     int uniquePaths(int m, int n) {
//         vector<vector<int>> dp(m, vector<int>(n, 0));
//         for (int i = 0; i < m; ++i) dp[i][0] = 1;
//         for (int j = 0; j < n; ++j) dp[0][j] = 1;

//         for (int i = 1; i < m; ++i) {
//             for (int j = 1; j < m; ++j){
//                 dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
//             }
//         }
//         return dp[m - 1][n - 1];
//     }
// };//�ҳ�����
class Solution {
public:
	int uniquePaths(int m, int n) {
		vector<vector<int>> dp(m, vector<int>(n, 0));
		for (int i = 0; i < m; ++i) dp[i][0] = 1;
		for (int j = 0; j < n; ++j) dp[0][j] = 1;

		for (int i = 1; i < m; ++i) {
			for (int j = 1; j < n; ++j){
				dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
			}
		}
		return dp[m - 1][n - 1];
	}
};


//746. ʹ����С������¥��
class Solution {
public:
	int minCostClimbingStairs(vector<int>& cost) {
		vector<int> dp(cost.size() + 1);
		dp[0] = cost[0];
		dp[1] = cost[1];

		int len_cost = cost.size();
		for (int i = 2; i < len_cost; ++i) { //���һ�ε��ﲻ�Ʒ�
			dp[i] = min(dp[i - 1], dp[i - 2]) + cost[i];
		}
		//return dp[len_cost];
		return min(dp[len_cost - 1], dp[len_cost - 2]); //���һ���ǲ��üƷѵ�
	}
};

//70. ��¥��
class Solution {
public:
	int climbStairs(int n) {
		if (n <= 2) return n;
		vector<int> dp(n + 1);
		dp[1] = 1;//0�㲻̫������ʵ
		dp[2] = 2;
		for (int i = 3; i <= n; ++i) {
			dp[i] = dp[i - 1] + dp[i - 2];
		}
		return dp[n];
	}
};

//509. 쳲�������
class Solution {
public:
	int fib(int N) {
		if (N <= 1) return N;
		int dp[2];
		dp[0] = 0;
		dp[1] = 1;
		int sum = 0;
		for (int i = 2; i <= N; ++i) {
			sum = dp[1] + dp[0];
			dp[0] = dp[1];
			dp[1] = sum;
		}
		return dp[1];
	}
};


//2021_3_31

//134. ����վ
class Solution {
public:
	int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
		int len_gas = gas.size();
		for (int i = 0; i < len_gas; ++i) {
			int rest = gas[i] - cost[i];
			//int index = (index + 1)%len_gas;
			int index = (i + 1) % len_gas;
			while (rest > 0 && index != i) {
				//rest = gas[index] - cost[index];
				rest += gas[index] - cost[index];
				index = (index + 1) % len_gas;
			}
			if (rest >= 0 && index == i) return i;
		}
		return -1;
	}
};

//1005. K ��ȡ������󻯵������
class Solution {
	static bool cmp(int a, int b) {
		return abs(a) > abs(b);
	}
public:
	int largestSumAfterKNegations(vector<int>& A, int K) {
		sort(A.begin(), A.end(), cmp);
		int len_A = A.size();
		for (int i = 0; i < len_A; ++i) {
			if (A[i] < 0 && K > 0) {
				A[i] *= -1;
				K--;
			}
		}
		while (K--) A[len_A - 1] *= -1;
		int result = 0;
		for (int a : A) result += a;
		return result;
	}
};

//45. ��Ծ��Ϸ II
class Solution {
public:
	int jump(vector<int>& nums) {
		int len_nums = nums.size();
		if (len_nums == 1) return 0;
		int cur_distance = 0;
		int next_distance = 0;
		int count = 0;

		for (int i = 0; i < len_nums - 1; ++i) { //�±��һ
			next_distance = max(nums[i] + i, next_distance);
			if (i == cur_distance) {
				count++;
				cur_distance = next_distance;
			}
		}
		return count;
	}
};

//55. ��Ծ��Ϸ
class Solution {
public:
	bool canJump(vector<int>& nums) {
		int cover = 0;
		if (nums.size() == 1) return true;
		int len_nums = nums.size();
		for (int i = 0; i <= cover; ++i) {
			cover = max(i + nums[i], cover);
			if (cover >= len_nums - 1) return true;
		}
		return false;
	}
};

//122. ������Ʊ�����ʱ�� II
class Solution {
public:
	int maxProfit(vector<int>& prices) {
		int result = 0;
		for (int i = 1; i < prices.size(); ++i) {
			result += max((prices[i] - prices[i - 1]), 0);
		}
		return result;
	}
};

//2021_3_30

//90. �Ӽ� II
class Solution {
private:
	vector<vector<int>> result;
	vector<int> path;
	void Trackbreaking(vector<int>& nums, int start_index, vector<bool>& used) {
		result.push_back(path);

		for (int i = start_index; i < nums.size(); ++i) {

			if (i > 0 && nums[i] == nums[i - 1] && used[i - 1] == false) continue; //
			path.push_back(nums[i]);
			used[i] = true;
			Trackbreaking(nums, i + 1, used);
			used[i] = false;
			path.pop_back();
		}
	}
public:
	vector<vector<int>> subsetsWithDup(vector<int>& nums) {
		result.clear();
		path.clear();
		vector<bool> used(nums.size(), false);
		sort(nums.begin(), nums.end());
		Trackbreaking(nums, 0, used);

		return result;
	}
};

//78. �Ӽ�
class Solution {
private:
	vector<vector<int>> result;
	vector<int> path;
	void Trackbreaking(vector<int>& nums, int start_index) {
		result.push_back(path);

		for (int i = start_index; i < nums.size(); i++) {
			path.push_back(nums[i]);
			Trackbreaking(nums, i + 1);
			path.pop_back();
		}
	}
public:
	vector<vector<int>> subsets(vector<int>& nums) {
		result.clear();
		path.clear();
		Trackbreaking(nums, 0);

		return result;
	}
};

//93. ��ԭ IP ��ַ
class Solution {
private:
	vector<string> result;
	void Trackbreaking(string &s, int start_index, int point_num) {

		if (point_num == 3) {
			if (isvalid(s, start_index, s.size() - 1)) {
				result.push_back(s);
			}
			return;
		}

		for (int i = start_index; i < s.size(); i++) {

			if (isvalid(s, start_index, i)) {
				s.insert(s.begin() + i + 1, '.');
				point_num++;
				Trackbreaking(s, i + 2, point_num);
				point_num--;
				s.erase(s.begin() + i + 1);
			}
		}
	}

	bool isvalid(string& s, int start, int end) {

		if (start > end) return false;
		if (s[start] == '0' && start != end) return false;

		int num = 0;
		for (int i = start; i <= end; i++) {

			if (s[i] > '9' || s[i] < '0') return false;
			num = num * 10 + (s[i] - '0');
			if (num > 255) return false;
		}
		return true;
	}

public:
	vector<string> restoreIpAddresses(string s) {
		result.clear();
		Trackbreaking(s, 0, 0);

		return result;
	}
};

//131. �ָ���Ĵ�
class Solution {
private:
	vector<vector<string>> result;
	vector<string> path;

	void Trackbreaking(string &s, int Startindex) {

		if (Startindex >= s.size()) {
			result.push_back(path);
			return;
		}

		for (int i = Startindex; i < s.size(); i++) {
			if (isPalindrome(s, Startindex, i)) {

				string str = s.substr(Startindex, i - Startindex + 1);
				path.push_back(str);
			}
			else {
				continue;
			}
			Trackbreaking(s, i + 1);
			path.pop_back();
		}

	}

	bool isPalindrome(const string& s, int start, int end) {
		for (int i = start, j = end; i < j; ++i, --j) {
			if (s[i] != s[j]) {
				return false;
			}
		}
		return true;
	}
public:
	vector<vector<string>> partition(string s) {

		result.clear();
		path.clear();

		Trackbreaking(s, 0);

		return result;
	}
};

//2021_3_29

//40. ����ܺ� II
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

//39. ����ܺ�
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
			Trackbreaking(candidates, target, sum, i);//���Startindex�Ӳ��ӵ�����
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

//17. �绰�������ĸ���
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

//216. ����ܺ� III
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

//77. ���
class Solution {
public:
	vector<vector<int>> result;
	vector<int> path;
	void TrachBreaking(int n, int k, int startIndex) {

		if (path.size() == k) {
			result.push_back(path);
			return;
		}

		for (int i = startIndex; i <= n - (k - path.size()) + 1; i++) { //�ٸ�����ȥ���ж�����
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

//538. �Ѷ���������ת��Ϊ�ۼ���
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

//108. ����������ת��Ϊ����������
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

//669. �޼�����������
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

//450. ɾ�������������еĽڵ�
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

//701. �����������еĲ������
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

//236. �������������������
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

//501. �����������е�����
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
				} //else if (pre != NULL || pre->val == cur->val) { //ɵ��
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

//530. ��������������С���Բ�
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

//61. ��ת����
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

//98. ��֤����������
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
	bool isValidBST(TreeNode* root) {//�������
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
		//st.push(root);//����

		while (cur != NULL || !st.empty()) {

			if (cur) {
				st.push(cur);
				cur = cur->left;
			}
			else {
				cur = st.top(); st.pop();
				//if (pre != NULL && cur->val >= cur->right->val) return false;
				if (pre != NULL && cur->val <= pre->val) return false;
				pre = cur;//����Ҫ
				cur = cur->right;
			}
		}
		return true;
	}
};

//700. �����������е�����
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
		while (root != NULL) {//�����ã��ͱ���
			if (root->val > val) root = root->left;
			else if (root->val < val) root = root->right;
			else return root;
		}
		return NULL;
	}
};


//617. �ϲ�������
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

//654. ��������
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

//105. ��ǰ��������������й��������
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
		//int inleftend = inbegin + range;//����
		//int inrightbegin = inbegin + range;
		int leftend = range;
		//int inrightbegin = range;����
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

//106. �����������������й��������
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
		int inleftend = range; //����ҿ�
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

//82. ɾ�����������е��ظ�Ԫ�� II
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

		while (cur->next && cur->next->next) {//��cur->next->next�����ֲ��������ǵ��ж�һ��cur->next->next�Ƿ�Ϊ��
			//while (cur->next) {//�յĽڵ��ǲ�������ֵ��

			if (cur->next->val == cur->next->next->val) {//��һ����֪

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

//��ָ Offer 18. ɾ������Ľڵ�
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

//83. ɾ�����������е��ظ�Ԫ��
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
				temp->next = NULL; //��ֹҰָ��
			}
			else {
				cur = cur->next;
			}
		}
		return head;
	}
};

//2021_3_24

//112. ·���ܺ�
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

//513. �������½ǵ�ֵ
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
	int findBottomLeftValue(TreeNode* root) {//�������yyds
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

//404. ��Ҷ��֮��
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

//100. ��ͬ����
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

//191. λ1�ĸ���
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
         int count = 1;//ѭ������һ��
         while (n =(n & (n-1))) count++;
         return count;
     }
 };

//222. ��ȫ�������Ľڵ����
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

//111. ����������С���
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

//559. N ������������
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

//104. ��������������
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

//101. �Գƶ�����
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

		queue<TreeNode*> que;//ջҲ����
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

//429. N �����Ĳ������
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

//637. �������Ĳ�ƽ��ֵ
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

//199. ������������ͼ
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

//107. �������Ĳ������ II
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

//102. �������Ĳ������
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


//145. �������ĺ������
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

//94. ���������������
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

//144. ��������ǰ�����
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

//239. �����������ֵ
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

//150. �沨�����ʽ��ֵ
class Solution {
public:
	int evalRPN(vector<string>& tokens) {
		stack<int> st;
		int tSize = tokens.size();

		for (int i = 0; i < tSize; i++) {
			//if (tokens[i] == '+' || tokens[i] == '-' || tokens[i] == '*' || tokens[i] == '/') {����
			if (tokens[i] == "+" || tokens[i] == "-" || tokens[i] == "*" || tokens[i] == "/") {
				int num1 = st.top();
				st.pop();
				int num2 = st.top();
				st.pop();
				if (tokens[i] == "+") st.push(num2 + num1);
				if (tokens[i] == "-") st.push(num2 - num1);//�⼸���ж���һ������Ĳ�����else if //����else if
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

//1047. ɾ���ַ����е����������ظ���
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
			result += st.top();//+=���������ٿ�
			st.pop();
		}
		reverse(result.begin(), result.end());
		return result;
	}
};

//20. ��Ч������
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

//225. �ö���ʵ��ջ
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

//232. ��ջʵ�ֶ���
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
//73. ��������
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
//28. ʵ�� strStr()
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
//242. ��Ч����ĸ��λ��
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

//142. �������� II
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
		//ListNode *fast = head->next;���ݹ�ʽ�պò�һ����Զ�޷�����
		ListNode *slow = head;

		while (fast && fast->next){
			fast = fast->next->next;
			slow = slow->next;
			//if (fast->val == slow->val){//ֵ���п����ظ���
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

//206. ��ת����
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

//203. �Ƴ�����Ԫ��
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

		while (head != NULL && head->val == val) { // ע�����ﲻ��if
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

//��ָ Offer 05. �滻�ո�
class Solution {
public:
	string replaceSpace(string s) {
		int oldSize = s.size();
		int count = 0;

		for (int i = 0; i < oldSize; i++){

			if (s[i] == ' ')
				count++;
		}

		//resize(s.size(), count * 2);//����
		s.resize(s.size() + count * 2);
		int newSize = s.size();

		//for (int i = newSize - 1, j = oldSize - 1; j >= 0; --i, --j){//�����벻�������
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

//541. ��ת�ַ��� II
// class Solution {
// public:
//     string reverseStr(string s, int k) {

//         for (int i = 0 ; i < s.size(); i+= (2 * k)){

//             if (i + k <= s.size()){//�Ⱥŵ����⣬����һ��k = size �������

//                 reverse(s.begin() + i, s.begin() + i + k);
//                 continue;
//             }

//             reverse(s.begin() + i, s.begin() + s.size());
//         }
//         return s;
//     }
// };//reverse �������÷�

class Solution {
public:

	void reverse(string &s, int start, int end){
		int offest = (end - start + 1) / 2; //start���±겻�ǹ̶�Ϊ0

		//for (start; start < (start + offest); start++, end--)//����д�ǡ�start + offest ���ж�������Զ������ֹ
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

//344. ��ת�ַ���
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

//59. �������� II
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

//1603. ���ͣ��ϵͳ
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

//209. ������С��������
// class Solution {
// public:
//     int minSubArrayLen(int target, vector<int>& nums) {
//         int result = INT32_MAX;//Ϊ�˵�һ�� len �ĸ�ֵ
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
//                     break;//���ѭ�����Խ�����
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

//27. �Ƴ�Ԫ��
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

//35. ��������λ��
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
//         int right = n - 1;//[1,3,5,6],7 ������������
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

//92. ��ת���� II
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
//925. ��������
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
//141. ��������
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

//88. �ϲ�������������
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

//344.��ת�ַ���
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

//26. ɾ�����������е��ظ���
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