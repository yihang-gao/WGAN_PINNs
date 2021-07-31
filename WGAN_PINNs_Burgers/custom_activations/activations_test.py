import tensorflow as tf
from custom_activations import group_sort


class MyTestCase(tf.test.TestCase):
    def test_group_sort(self):
        x = tf.constant([[6, 5, 8, 7, 2, 1, 4, 3], [2, 3, 4, 5, 6, 7, 8, 9]])
        ans = tf.constant([[5, 6, 7, 8, 1, 2, 3, 4], [2, 3, 4, 5, 6, 7, 8, 9]])

        ret = group_sort(x, group_size=2)

        self.assertAllEqual(ret, ans)


if __name__ == '__main__':
    tf.test.main()
