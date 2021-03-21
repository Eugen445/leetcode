#include<iostream>
using namespace std;

//

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