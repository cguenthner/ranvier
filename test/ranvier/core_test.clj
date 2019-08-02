(ns ranvier.core-test
  (:require [clojure.test :refer [deftest is testing]]
            [ranvier.core :as r :refer :all :exclude [min max]]
            [ranvier.test-utils :refer :all]
            [tensure.core :as m]))

(refer-privates 'ranvier.core :reverse-broadcast :add-2 :sub-2 :mul-2 :div-2 :max-2
                :min-2 :size-axes :eq-2 :ne-2 :and-2 :or-2 :gt-2 :ge-2 :lt-2 :le-2 :join-2-along
                :shift-all-by :shift-dim-by)

(def pi Math/PI)

(deftest reverse-broadcast-test
  (let [a-data [[1 2] [3 4]]
        b-data [[5 6] [7 8]]
        c-data [a-data b-data]
        a (m/array a-data)
        b (m/array b-data)
        c (m/array c-data)
        d (m/array [c-data c-data])]
    (testing "it works correctly when the target is a scalar"
      (is (nd= (reverse-broadcast (m/array 69) nil) 69))
      (is (nd= (reverse-broadcast (m/array [1 2 3]) nil) 6))
      (is (nd= (reverse-broadcast (m/array [1 2 3]) [1]) [6]))
      (is (nd= (reverse-broadcast (m/array [[1 2] [3 4]]) nil) 10))
      (is (nd= (reverse-broadcast (m/array [[[1 2] [3 4]] [[5 6] [7 8]]]) nil) 36)))
    (testing "it works correctly when the target is a vector"
      (is (nd= (reverse-broadcast (m/array [1 2]) [2]) [1 2]))
      (is (nd= (reverse-broadcast (m/array [[1 2]]) [2]) [1 2]))
      (is (nd= (reverse-broadcast (m/array [[1 2] [3 4]]) [2]) [4 6]))
      (is (nd= (reverse-broadcast (m/array [[[1 2] [3 4]] [[5 6] [7 8]]]) [2]) [16 20]))
      (is (nd= (reverse-broadcast (m/array [[1] [2]]) [2]) [1 2]))
      (is (nd= (reverse-broadcast (m/array [[1 2 3] [4 5 6]]) [3]) [5 7 9]))
      (is (nd= (reverse-broadcast (m/array [[1 2 3] [4 5 6]]) [2]) [6 15]))
      (is (nd= (reverse-broadcast (m/array [[[1 2 3] [4 5 6]] [[7 8 9] [10 11 12]]]) [3]) [22 26 30])))
    (testing "it works correctly when the target is a matrix"
      (is (nd= (reverse-broadcast a [2 2]) a))
      (is (nd= (reverse-broadcast c [2 2]) [[6 8] [10 12]]))
      (is (nd= (reverse-broadcast d [2 2]) [[12 16] [20 24]]))
      (is (nd= (reverse-broadcast (m/array [[1 2]]) [1 1]) [[3]]))
      (is (nd= (reverse-broadcast (m/array [[1] [2]]) [1 1]) [[3]]))
      (is (nd= (reverse-broadcast (m/array [[1 2 3] [4 5 6]]) [1 3]) [[5 7 9]]))
      (is (nd= (reverse-broadcast (m/array [[1 2 3] [4 5 6]]) [2 1]) [[6] [15]]))
      (is (nd= (reverse-broadcast (m/array [[[1 2 3 4]
                                             [5 6 7 8]]
                                            [[4 5 1 2]
                                             [5 7 6 8]]]) [2 4]) [[5 7 4 6] [10 13 13 16]])))
    (testing "it works correctly when the target is an nd-array"
      (is (nd= (reverse-broadcast c [2 2 2]) c))
      (is (nd= (reverse-broadcast d [2 2 2]) [[[2 4] [6 8]] [[10 12] [14 16]]]))
      (is (nd= (reverse-broadcast (m/array [[[1 2]]]) [1 1 1]) [[[3]]]))
      (is (nd= (reverse-broadcast (m/array [[[1] [2]]]) [1 1 1]) [[[3]]]))
      (is (nd= (reverse-broadcast (m/array [[[[1 2 3] [4 5 6]] [[2 1 3] [6 5 4]]]
                                            [[[8 9 6] [0 1 4]] [[5 5 5] [7 3 2]]]]) [2 2 3])
               [[[9 11 9] [4 6 10]] [[7 6 8] [13 8 6]]]))
      (is (nd= (reverse-broadcast (m/array [[[[1 2 3] [4 5 6]] [[2 1 3] [6 5 4]]]
                                            [[[8 9 6] [0 1 4]] [[5 5 5] [7 3 2]]]]) [1 1 3])
               [[[33 31 33]]])))
    (testing "it throws an exception when shapes are incompatible"
      (is (thrown? Exception (reverse-broadcast (m/array 69) [1]))))))

(deftest add-2-test
  (testing "scalar + scalar"
    (let [[v [da db]] (test-op add-2 2 3 4)]
      (is (nd= v 7))
      (is (nd= da 2))
      (is (nd= db 2))))
  (testing "vector + scalar"
    (let [[v [da db]] (test-op add-2 [2 1] [1 3] 4)]
      (is (nd= v [5 7]))
      (is (nd= da [2 1]))
      (is (nd= db 3))))
  (testing "matrix + scalar"
    (let [[v [da db]] (test-op add-2 [[5 6] [7 8]] [[1 2] [3 4]] 2)]
      (is (nd= v [[3 4] [5 6]]))
      (is (nd= da [[5 6] [7 8]]))
      (is (nd= db 26))))
  (testing "tensor + scalar"
    (let [[v [da db]] (test-op add-2 [[[1 2] [2 1]] [[1 1] [2 2]]] [[[1 2] [3 4]] [[5 6] [7 8]]] 2)]
      (is (nd= v [[[3 4] [5 6]] [[7 8] [9 10]]]))
      (is (nd= da [[[1 2] [2 1]] [[1 1] [2 2]]]))
      (is (nd= db 12))))
  (testing "vector + vector"
    (let [[v [da db]] (test-op add-2 [2 1] [1 3] [5 7])]
      (is (nd= v [6 10]))
      (is (nd= da [2 1]))
      (is (nd= db [2 1]))))
  (testing "vector + matrix"
    (let [[v [da db]] (test-op add-2 [[1 2] [3 4]] [1 3] [[5 7] [9 11]])]
      (is (nd= v [[6 10] [10 14]]))
      (is (nd= da [4 6]))
      (is (nd= db [[1 2] [3 4]]))))
  (testing "matrix + column vector"
    (let [[v [da db]] (test-op add-2 [[0 3 1] [2 1 4]] [[1 2 3] [4 5 6]] [7 8])]
      (is (nd= v [[8 9 10] [12 13 14]]))
      (is (nd= da [[0 3 1] [2 1 4]]))
      (is (nd= db [4 7]))))
  (testing "matrix + column matrix"
    (let [[v [da db]] (test-op add-2 [[0 3 1] [2 1 4]] [[1 2 3] [4 5 6]] [[7] [8]])]
      (is (nd= v [[8 9 10] [12 13 14]]))
      (is (nd= da [[0 3 1] [2 1 4]]))
      (is (nd= db [[4] [7]]))))
  (testing "matrix + row vector"
    (let [[v [da db]] (test-op add-2 [[0 3 1] [2 1 4]] [[1 2 3] [4 5 6]] [7 8 9])]
      (is (nd= v [[8 10 12] [11 13 15]]))
      (is (nd= da [[0 3 1] [2 1 4]]))
      (is (nd= db [2 4 5]))))
  (testing "matrix + row matrix"
    (let [[v [da db]] (test-op add-2 [[0 3 1] [2 1 4]] [[1 2 3] [4 5 6]] [[7 8 9]])]
      (is (nd= v [[8 10 12] [11 13 15]]))
      (is (nd= da [[0 3 1] [2 1 4]]))
      (is (nd= db [[2 4 5]]))))
  (testing "vector + tensor"
    (let [[v [da db]] (test-op add-2 [[[1 2] [2 1]] [[1 1] [2 2]]] [1 3] [[[1 2] [3 4]] [[5 6] [7 8]]])]
      (is (nd= v [[[2 5] [4 7]] [[6 9] [8 11]]]))
      (is (nd= da [6 6]))
      (is (nd= db [[[1 2] [2 1]] [[1 1] [2 2]]]))))
  (testing "matrix + matrix"
    (let [[v [da db]] (test-op add-2 [[1 3] [5 7]] [[1 2] [3 4]] [[5 6] [7 8]])]
      (is (nd= v [[6 8] [10 12]]))
      (is (nd= da [[1 3] [5 7]]))
      (is (nd= db [[1 3] [5 7]]))))
  (testing "matrix + tensor"
    (let [[v [da db]] (test-op add-2 [[[1 2] [2 1]] [[1 1] [2 2]]] [[1 3] [5 7]] [[[1 2] [3 4]] [[5 6] [7 8]]])]
      (is (nd= v [[[2 5] [8 11]] [[6 9] [12 15]]]))
      (is (nd= da [[2 3] [4 3]]))
      (is (nd= db [[[1 2] [2 1]] [[1 1] [2 2]]]))))
  (testing "tensor + tensor"
    (let [[v [da db]] (test-op add-2
                               [[[1 2] [2 1]] [[1 1] [2 2]]]
                               [[[7 8] [5 6]] [[3 4] [1 2]]]
                               [[[1 2] [3 4]] [[5 6] [7 8]]])]
      (is (nd= v [[[8 10] [8 10]] [[8 10] [8 10]]]))
      (is (nd= da [[[1 2] [2 1]] [[1 1] [2 2]]]))
      (is (nd= db [[[1 2] [2 1]] [[1 1] [2 2]]])))))

(deftest sub-2-test
  (testing "scalar - scalar"
    (let [[v [da db]] (test-op sub-2 2 3 4)]
      (is (nd= v -1))
      (is (nd= da 2))
      (is (nd= db -2))))
  (testing "vector - scalar"
    (let [[v [da db]] (test-op sub-2 [2 6] [1 7] 4)]
      (is (nd= v [-3 3]))
      (is (nd= da [2 6]))
      (is (nd= db -8))))
  (testing "matrix - scalar"
    (let [[v [da db]] (test-op sub-2 [[5 6] [7 8]] [[1 2] [3 4]] 2)]
      (is (nd= v [[-1 0] [1 2]]))
      (is (nd= da [[5 6] [7 8]]))
      (is (nd= db -26))))
  (testing "tensor - scalar"
    (let [[v [da db]] (test-op sub-2 [[[1 2] [2 1]] [[1 1] [2 2]]] [[[7 8] [5 6]] [[3 4] [1 2]]] 3)]
      (is (nd= v [[[4 5] [2 3]] [[0 1] [-2 -1]]]))
      (is (nd= da [[[1 2] [2 1]] [[1 1] [2 2]]]))
      (is (nd= db -12))))
  (testing "vector - vector"
    (let [[v [da db]] (test-op sub-2 [2 6] [1 7] [20 4])]
      (is (nd= v [-19 3]))
      (is (nd= da [2 6]))
      (is (nd= db [-2 -6]))))
  (testing "vector - matrix"
    (let [[v [da db]] (test-op sub-2 [[5 6] [7 8]] [1 7] [[1 2] [3 4]])]
      (is (nd= v [[0 5] [-2 3]]))
      (is (nd= da [12 14]))
      (is (nd= db [[-5 -6] [-7 -8]]))))
  (testing "column vector - matrix"
    (let [[v [da db]] (test-op sub-2 [[0 3 1] [2 1 4]] [3 4] [[1 2 3] [4 5 6]])]
      (is (nd= v [[2 1 0] [0 -1 -2]]))
      (is (nd= da [4 7]))
      (is (about= db [[0 -3 -1] [-2 -1 -4]]))))
  (testing "vector - tensor"
    (let [[v [da db]] (test-op sub-2 [[[1 2] [2 1]] [[1 1] [2 2]]] [1 7] [[[1 2] [3 4]] [[4 3] [2 1]]])]
      (is (nd= v [[[0 5] [-2 3]] [[-3 4] [-1 6]]]))
      (is (nd= da [6 6]))
      (is (nd= db [[[-1 -2] [-2 -1]] [[-1 -1] [-2 -2]]]))))
  (testing "matrix - column vector"
    (let [[v [da db]] (test-op sub-2 [[0 3 1] [2 1 4]] [[1 2 3] [4 5 6]] [3 4])]
      (is (nd= v [[-2 -1 0] [0 1 2]]))
      (is (nd= da [[0 3 1] [2 1 4]]))
      (is (nd= db [-4 -7]))))
  (testing "matrix - row vector"
    (let [[v [da db]] (test-op sub-2 [[0 3 1] [2 1 4]] [[1 2 3] [4 5 6]] [5 -2 0])]
      (is (nd= v [[-4 4 3] [-1 7 6]]))
      (is (nd= da [[0 3 1] [2 1 4]]))
      (is (nd= db [-2 -4 -5]))))
  (testing "matrix - column matrix"
    (let [[v [da db]] (test-op sub-2 [[0 3 1] [2 1 4]] [[1 2 3] [4 5 6]] [[3] [4]])]
      (is (nd= v [[-2 -1 0] [0 1 2]]))
      (is (nd= da [[0 3 1] [2 1 4]]))
      (is (nd= db [[-4] [-7]]))))
  (testing "column matrix - matrix"
    (let [[v [da db]] (test-op sub-2 [[0 3 1] [2 1 4]] [[3] [4]] [[1 2 3] [4 5 6]])]
      (is (nd= v [[2 1 0] [0 -1 -2]]))
      (is (nd= da [[4] [7]]))
      (is (about= db [[0 -3 -1] [-2 -1 -4]]))))
  (testing "matrix - row matrix"
    (let [[v [da db]] (test-op sub-2 [[0 3 1] [2 1 4]] [[1 2 3] [4 5 6]] [[5 -2 0]])]
      (is (nd= v [[-4 4 3] [-1 7 6]]))
      (is (nd= da [[0 3 1] [2 1 4]]))
      (is (nd= db [[-2 -4 -5]]))))
  (testing "matrix - matrix"
    (let [[v [da db]] (test-op sub-2 [[2 3] [5 4]] [[1 2] [3 4]] [[5 6] [2 1]])]
      (is (nd= v [[-4 -4] [1 3]]))
      (is (nd= da [[2 3] [5 4]]))
      (is (nd= db [[-2 -3] [-5 -4]]))))
  (testing "tensor - matrix"
    (let [[v [da db]] (test-op sub-2 [[[1 2] [-2 1]] [[1 -1] [2 2]]] [[[1 2] [3 4]] [[5 6] [7 8]]] [[5 6] [2 1]])]
      (is (nd= v [[[-4 -4] [1 3]] [[0 0] [5 7]]]))
      (is (nd= da [[[1 2] [-2 1]] [[1 -1] [2 2]]]))
      (is (nd= db [[-2 -1] [0 -3]]))))
  (testing "tensor - matrix"
    (let [[v [da db]] (test-op sub-2
                               [[[1 2] [-2 1]] [[1 -1] [2 2]]]
                               [[[1 2] [3 4]] [[5 6] [7 8]]]
                               [[[7 8] [5 6]] [[3 4] [1 2]]])]
      (is (nd= v [[[-6 -6] [-2 -2]] [[2 2] [6 6]]]))
      (is (nd= da [[[1 2] [-2 1]] [[1 -1] [2 2]]]))
      (is (nd= db [[[-1 -2] [2 -1]] [[-1 1] [-2 -2]]])))))

(deftest mul-2-test
  (testing "scalar * scalar"
    (let [[v [da db]] (test-op mul-2 2 3 4)]
      (is (nd= v 12))
      (is (nd= da 8))
      (is (nd= db 6))))
  (testing "vector * scalar"
    (let [[v [da db]] (test-op mul-2 [4 5] [1 2] 3)]
      (is (nd= v [3 6]))
      (is (nd= da [12 15]))
      (is (nd= db 14))))
  (testing "matrix * scalar"
    (let [[v [da db]] (test-op mul-2 [[5 6] [7 8]] [[1 2] [3 4]] 2)]
      (is (nd= v [[2 4] [6 8]]))
      (is (nd= da [[10 12] [14 16]]))
      (is (nd= db 70))))
  (testing "tensor * scalar"
    (let [[v [da db]] (test-op mul-2 [[[1 2] [2 1]] [[1 1] [2 2]]] [[[1 2] [3 4]] [[5 6] [7 8]]] 2)]
      (is (nd= v [[[2 4] [6 8]] [[10 12] [14 16]]]))
      (is (nd= da [[[2 4] [4 2]] [[2 2] [4 4]]]))
      (is (nd= db 56))))
  (testing "vector * vector"
    (let [[v [da db]] (test-op mul-2 [5 6] [1 2] [3 4])]
      (is (nd= v [3 8]))
      (is (nd= da [15 24]))
      (is (nd= db [5 12]))))
  (testing "matrix * vector"
    (let [[v [da db]] (test-op mul-2 [[7 8] [9 10]] [[1 2] [3 4]] [5 6])]
      (is (nd= v [[5 12] [15 24]]))
      (is (nd= da [[35 48] [45 60]]))
      (is (nd= db [34 56]))))
  (testing "tensor * vector"
    (let [[v [da db]] (test-op mul-2 [[[1 2] [2 1]] [[1 1] [2 2]]] [[[1 2] [3 4]] [[5 6] [7 8]]] [2 3])]
      (is (nd= v [[[2 6] [6 12]] [[10 18] [14 24]]]))
      (is (nd= da [[[2 6] [4 3]] [[2 3] [4 6]]]))
      (is (nd= db [26 30]))))
  (testing "matrix * matrix"
    (let [[v [da db]] (test-op mul-2 [[4 3] [2 1]] [[1 2] [3 4]] [[5 6] [7 8]])]
      (is (nd= v [[5 12] [21 32]]))
      (is (nd= da [[20 18] [14 8]]))
      (is (nd= db [[4 6] [6 4]]))))
  (testing "matrix * column vector"
    (let [[v [da db]] (test-op mul-2 [[0 3 1] [2 1 4]] [[1 2 3] [4 5 6]] [2 3])]
      (is (nd= v [[2 4 6] [12 15 18]]))
      (is (nd= da [[0 6 2] [6 3 12]]))
      (is (nd= db [9 37]))))
  (testing "matrix * column matrix"
    (let [[v [da db]] (test-op mul-2 [[0 3 1] [2 1 4]] [[1 2 3] [4 5 6]] [[2] [3]])]
      (is (nd= v [[2 4 6] [12 15 18]]))
      (is (nd= da [[0 6 2] [6 3 12]]))
      (is (nd= db [[9] [37]]))))
  (testing "matrix * row vector"
    (let [[v [da db]] (test-op mul-2 [[0 3 1] [2 1 4]] [3 2 4] [[1 2 3] [4 5 6]])]
      (is (nd= v [[3 4 12] [12 10 24]]))
      (is (nd= da [8 11 27]))
      (is (nd= db [[0 6 4] [6 2 16]]))))
  (testing "matrix * row matrix"
    (let [[v [da db]] (test-op mul-2 [[0 3 1] [2 1 4]] [[3 2 4]] [[1 2 3] [4 5 6]])]
      (is (nd= v [[3 4 12] [12 10 24]]))
      (is (nd= da [[8 11 27]]))
      (is (nd= db [[0 6 4] [6 2 16]]))))
  (testing "matrix * tensor"
    (let [[v [da db]] (test-op mul-2 [[[1 2] [2 1]] [[1 1] [2 2]]] [[1 2] [3 4]]
                               [[[1 2] [3 4]] [[5 6] [7 8]]])]
      (is (nd= v [[[1 4] [9 16]] [[5 12] [21 32]]]))
      (is (nd= da [[6 10] [20 20]]))
      (is (nd= db [[[1 4] [6 4]] [[1 2] [6 8]]]))))
  (testing "tensor * tensor"
    (let [[v [da db]] (test-op mul-2
                               [[[1 2] [2 1]] [[1 1] [2 2]]]
                               [[[1 2] [3 4]] [[5 6] [7 8]]]
                               [[[3 4] [2 1]] [[4 3] [1 2]]])]
      (is (nd= v [[[3 8] [6 4]] [[20 18] [7 16]]]))
      (is (nd= da [[[3 8] [4 1]] [[4 3] [2 4]]]))
      (is (nd= db [[[1 4] [6 4]] [[5 6] [14 16]]])))))

(deftest div-2-test
  (let [g (G (div-2 :a :b))
        check (fn [tolerance a b expected-result]
                (is (about= (evaluate g {:a a :b b}) expected-result))
                (numerically-validated? g {:a a :b b} tolerance))]
    (testing "scalar / scalar"
      (check 1e-3 8 2 4)
      (check 1e-4 1 8 1/8))
    (testing "vector / scalar"
      (check 1e-2 [1 2 3] 5 [1/5 2/5 3/5])
      (check 1e-2 [1 2 3] 0.5 [2 4 6]))
    (testing "scalar / vector"
      (check 1e-3 5 [1 2 3] [5 2.5 5/3]))
    (testing "matrix / scalar"
      (check 1e-2 [[1 2] [3 4]] 10 [[0.1 0.2] [0.3 0.4]]))
    (testing "scalar / matrix"
      (check 1e-2 1 [[1 2] [3 4]] [[1 1/2] [1/3 1/4]]))
    (testing "tensor / scalar"
      (check 1e-2 [[[1 2] [3 4]] [[5 6] [7 8]]] 6 [[[1/6 1/3] [1/2 2/3]] [[5/6 1] [7/6 4/3]]]))
    (testing "scalar / tensor"
      (check 1e-2 3 [[[1 2] [3 4]] [[5 6] [7 8]]] [[[3 3/2] [1 3/4]] [[3/5 1/2] [3/7 3/8]]]))
    (testing "vector / vector"
      (check 1e-2 [1 2 3] [4 5 6] [1/4 2/5 1/2]))
    (testing "matrix / vector"
      (check 1e-2 [[1 2 3] [4 5 6]] [1 2 3] [[1 1 1] [4 2.5 2]])
      (check 0.1 [[1 2 3] [4 5 6]] [0.5 0.1 10] [[2 20 0.3] [8 50 0.6]]))
    (testing "matrix / column vector"
      (check 1e-2 [[1 2 3] [4 5 6]] [0.1 2] [[10 20 30] [2 2.5 3]]))
    (testing "matrix / column matrix"
      (check 1e-2 [[1 2 3] [4 5 6]] [[0.1] [2]] [[10 20 30] [2 2.5 3]]))
    (testing "matrix / row vector"
      (check 0.01 [[1 2 3] [4 5 6]] [0.1 2 0.2] [[10 1 15] [40 2.5 30]]))
    (testing "matrix / row matrix"
      (check 0.01 [[1 2 3] [4 5 6]] [[0.1 2 0.2]] [[10 1 15] [40 2.5 30]]))
    (testing "vector / matrix"
      (check 0.1 [1 2 3] [[0.5 0.2 0.1] [2 3 4]] [[2 10 30] [1/2 2/3 3/4]]))
    (testing "tensor / vector"
      (check 0.01 [[[1 2] [3 4]] [[5 6] [7 8]]] [2 0.1] [[[0.5 20] [1.5 40]] [[2.5 60] [3.5 80]]]))
    (testing "vector / tensor"
      (check 0.1 [2 0.1] [[[1 2] [3 4]] [[5 6] [7 8]]] [[[2 0.05] [2/3 0.025]] [[2/5 1/60] [2/7 1/80]]]))
    (testing "matrix / matrix"
      (check 1e-2 [[1 2] [7 8]] [[5 6] [3 4]] [[0.2 1/3] [7/3 2]]))
    (testing "tensor / matrix"
      (check 0.1 [[[1 2] [3 4]] [[5 6] [7 8]]] [[20 10] [0.5 0.25]]
             [[[0.05 0.2] [6 16]] [[0.25 0.6] [14 32]]]))
    (testing "matrix / tensor"
      (check 0.1 [[20 10] [1 2]] [[[1 2] [3 4]] [[5 6] [7 8]]]
             [[[20 5] [1/3 0.5]] [[4 5/3] [1/7 1/4]]]))
    (testing "tensor / tensor"
      (check 0.2 [[[1 2] [3 4]] [[5 6] [7 8]]]
             [[[10 4] [3 0.5]] [[1/3 18] [3 20]]]
             [[[0.1 0.5] [1 8]] [[15 1/3] [7/3 0.4]]]))))

(deftest pow-test
  (testing "scalar ^ scalar"
    (let [[v [da db]] (test-op pow 10 3 4)]
      (is (nd= v 81))
      (is (nd= da 1080))
      (is (about= db 889.8759539))))
  (testing "matrix ^ scalar"
    (let [[v [da db]] (test-op pow [[2 1] [-1 2]] [[3 1] [1 3]] 4)]
      (is (nd= v [[81 1] [1 81]]))
      (is (nd= da [[216 4] [-4 216]]))
      (is (about= db 355.950381))))
  (testing "scalar ^ matrix"
    (let [[v [da db]] (test-op pow [[2 1] [-1 2]] 3 [[0 1] [2 3]])]
      (is (nd= v [[1 3] [9 27]]))
      (is (nd= da 49))
      (is (about= db [[2.197225 3.295837] [-9.887511 59.3250636]])))))

(deftest mmul-test
  (testing "scalar * scalar"
    (let [[v [da db]] (test-op mmul 2 3 4)]
      (is (nd= v 12))
      (is (nd= da 8))
      (is (nd= db 6))))
  (testing "vector * scalar"
    (let [[v [da db]] (test-op mmul [4 5] [1 2] 3)]
      (is (nd= v [3 6]))
      (is (nd= da [12 15]))
      (is (nd= db 14))))
  (testing "matrix * scalar"
    (let [[v [da db]] (test-op mmul [[5 6] [7 8]] [[1 2] [3 4]] 2)]
      (is (nd= v [[2 4] [6 8]]))
      (is (nd= da [[10 12] [14 16]]))
      (is (nd= db 70))))
  (testing "tensor * scalar"
    (let [[v [da db]] (test-op mmul [[[1 2] [2 1]] [[1 1] [2 2]]] [[[1 2] [3 4]] [[5 6] [7 8]]] 2)]
      (is (nd= v [[[2 4] [6 8]] [[10 12] [14 16]]]))
      (is (nd= da [[[2 4] [4 2]] [[2 2] [4 4]]]))
      (is (nd= db 56))))
  (testing "vector . vector"
    (let [[v [da db]] (test-op mmul 2 [1 2 3] [4 5 6])]
      (is (nd= v 32))
      (is (nd= da [8 10 12]))
      (is (nd= db [2 4 6]))))
  (testing "matrix . vector"
    (let [[v [da db]] (test-op mmul [2 3] [[1 2 3] [4 5 6]] [2 1 3])]
      (is (nd= v [13 31]))
      (is (nd= da [[4 2 6] [6 3 9]]))
      (is (nd= db [14 19 24]))))
  (testing "vector . matrix"
    (let [[v [da db]] (test-op mmul [2 3] [2 1 3] [[1 2] [3 4] [5 6]])]
      (is (nd= v [20 26]))
      (is (nd= da [8 18 28]))
      (is (nd= db [[4 6] [2 3] [6 9]]))))
  (testing "tensor . vector"
    (let [[v [da db]] (test-op mmul [[1 2] [2 1]] [[[1 2 3] [4 5 6]] [[2 3 1] [6 5 4]]] [2 1 3])]
      (is (nd= v [[13 31] [10 29]]))
      (is (nd= da [[[2 1 3] [4 2 6]] [[4 2 6] [2 1 3]]]))
      (is (nd= db [19 23 21]))))
  (testing "vector . tensor"
    (let [[v [da db]] (test-op mmul [[1 2] [2 1]] [2 1 3] [[[1 2] [4 6]] [[2 3] [5 5]] [[3 1] [6 4]]])]
      (is (nd= v [[13 10] [31 29]]))
      (is (nd= da [19 23 21]))
      (is (nd= db [[[2 4] [4 2]] [[1 2] [2 1]] [[3 6] [6 3]]]))))
  (testing "matrix . matrix"
    (let [[v [da db]] (test-op mmul [[2 3]
                                     [1 2]] [[1 2 3]
                                             [4 5 6]]
                               [[0 2]
                                [3 1]
                                [2 4]])]
      (is (nd= v [[12 16] [27 37]]))
      (is (nd= da [[6 9 16] [4 5 10]]))
      (is (nd= db [[6 11] [9 16] [12 21]]))))
  (testing "matrix . tensor"
    (let [[v [da db]] (test-op mmul [[[2 2] [1 1]]
                                     [[1 2] [2 1]]] [[2 1 3]
                                                     [1 1 2]] [[[1 2] [4 6]]
                                                               [[2 3] [5 5]]
                                                               [[3 1] [6 4]]])]
      (is (nd= v [[[13 10] [31 29]] [[9 7] [21 19]]]))
      (is (nd= da [[16 20 18] [19 23 21]]))
      (is (nd= db [[[5 6] [4 3]] [[3 4] [3 2]] [[8 10] [7 5]]]))))
  (testing "tensor . matrix"
    (let [[v [da db]] (test-op mmul [[[2 2]
                                      [1 1]]
                                     [[1 2]
                                      [2 1]]]
                               [[[1 2 3]
                                 [4 5 6]]
                                [[2 3 1]
                                 [6 5 4]]] [[2 1]
                                            [1 1]
                                            [3 2]])]
      (is (nd= v [[[13 9] [31 21]] [[10 7] [29 19]]]))
      (is (nd= da [[[6 4 10] [3 2 5]] [[4 3 7] [5 3 8]]]))
      (is (nd= db [[20 16] [22 20] [21 18]]))))
  (testing "tensor . tensor"
    (let [[v [da db]] (test-op mmul
                               [[[[1 1] [2 2]] [[1 2] [2 1]]]
                                [[[2 1] [1 2]] [[2 2] [1 1]]]]
                               [[[1 0 1]
                                 [2 1 3]]
                                [[0 2 0]
                                 [3 1 1]]] [[[1 2] [2 1]]
                                            [[1 1] [1 2]]
                                            [[2 1] [1 1]]])]
      (is (nd= v [[[[3 3] [3 2]] [[9 8] [8 7]]]
                  [[[2 2] [2 4]] [[6 8] [8 6]]]]))
      (is (nd= da [[[9 8 7] [10 7 7]] [[8 8 8] [9 7 8]]]))
      (is (nd= db [[[9 11] [9 7]]
                   [[7 6] [5 6]]
                   [[6 9] [9 6]]])))))

; This works but takes a while to run and can potential exhaust heap space for large matrices.
; Move this to a separate namespace when there are more generative tests.
; It would also be better to sample shapes strategically instead of randomly.
#_(deftest mmul-generative-test
    (dotimes [_ 10]
      (let [max-dimensionality 3
            max-dimension-size 4
            a-shape (random-shape 0 max-dimensionality max-dimension-size)
            common-shape (take-last 2 a-shape)
            b-shape (concat (reverse common-shape)
                            (random-shape 0 (- max-dimensionality (count common-shape)) max-dimension-size)
                            (repeatedly (rand-int 8) #(inc (rand-int 5))))
            a (m/sample-uniform a-shape)
            b (m/sample-uniform b-shape)]
        (numerically-validated? (mmul :a :b) {:a a :b b} 1e-2))))

(deftest max-2-test
  (testing "max(scalar, scalar)"
    (let [[v [da db]] (test-op max-2 3 1 2)]
      (is (nd= v 2))
      (is (nd= da 0))
      (is (nd= db 3)))
    (let [[v [da db]] (test-op max-2 3 2 1)]
      (is (nd= v 2))
      (is (nd= da 3))
      (is (nd= db 0)))
    (let [[v [da db]] (test-op max-2 3 2 2)]
      (is (nd= v 2))
      (is (nd= da 3))
      (is (nd= db 3))))
  (testing "max(matrix, scalar)"
    (let [[v [da db]] (test-op max-2 [[1 2] [3 4]] [[0 2] [1 3]] 1)]
      (is (nd= v [[1 2] [1 3]]))
      (is (nd= da [[0 2] [3 4]]))
      (is (nd= db 4)))
    (let [[v [da db]] (test-op max-2 [[1 2] [3 4]] 1 [[0 2] [1 3]])]
      (is (nd= v [[1 2] [1 3]]))
      (is (nd= da 4))
      (is (nd= db [[0 2] [3 4]]))))
  (testing "max(matrix, column vector)"
    (let [[v [da db]] (test-op max-2 [[7 3 1] [2 1 4]] [[1 2 3] [4 5 6]] [2 4])]
      (is (nd= v [[2 2 3] [4 5 6]]))
      (is (nd= da [[0 3 1] [2 1 4]]))
      (is (nd= db [10 2]))))
  (testing "max(matrix, column matrix)"
    (let [[v [da db]] (test-op max-2 [[7 3 1] [2 1 4]] [[1 2 3] [4 5 6]] [[2] [4]])]
      (is (nd= v [[2 2 3] [4 5 6]]))
      (is (nd= da [[0 3 1] [2 1 4]]))
      (is (nd= db [[10] [2]]))))
  (testing "max(matrix, row vector)"
    (let [[v [da db]] (test-op max-2 [[7 3 1] [2 1 4]] [0 4 9] [[1 2 3] [4 5 6]])]
      (is (nd= v [[1 4 9] [4 5 9]]))
      (is (nd= da [0 3 5]))
      (is (nd= db [[7 0 0] [2 1 0]]))))
  (testing "max(matrix, row matrix)"
    (let [[v [da db]] (test-op max-2 [[7 3 1] [2 1 4]] [[0 4 9]] [[1 2 3] [4 5 6]])]
      (is (nd= v [[1 4 9] [4 5 9]]))
      (is (nd= da [[0 3 5]]))
      (is (nd= db [[7 0 0] [2 1 0]]))))
  (testing "max(matrix, matrix)"
    (let [[v [da db]] (test-op max-2 [[1 2] [3 4]] [[0 3] [2 6]] [[1 2] [3 0]])]
      (is (nd= v [[1 3] [3 6]]))
      (is (nd= da [[0 2] [0 4]]))
      (is (nd= db [[1 0] [3 0]])))))

(deftest min-2-test
  (testing "min(scalar, scalar)"
    (let [[v [da db]] (test-op min-2 3 1 2)]
      (is (nd= v 1))
      (is (nd= da 3))
      (is (nd= db 0))))
  (testing "min(matrix, scalar)"
    (let [[v [da db]] (test-op min-2 [[1 2] [3 4]] [[0 2] [1 3]] 1)]
      (is (nd= v [[0 1] [1 1]]))
      (is (nd= da [[1 0] [3 0]]))
      (is (nd= db 9))))
  (testing "min(matrix, column vector)"
    (let [[v [da db]] (test-op min-2 [[7 3 1] [2 1 4]] [2 4] [[1 2 3] [4 5 6]])]
      (is (nd= v [[1 2 2] [4 4 4]]))
      (is (nd= da [4 7]))
      (is (nd= db [[7 3 0] [2 0 0]]))))
  (testing "min(matrix, column matrix)"
    (let [[v [da db]] (test-op min-2 [[7 3 1] [2 1 4]] [[2] [4]] [[1 2 3] [4 5 6]])]
      (is (nd= v [[1 2 2] [4 4 4]]))
      (is (nd= da [[4] [7]]))
      (is (nd= db [[7 3 0] [2 0 0]]))))
  (testing "min(matrix, row vector)"
    (let [[v [da db]] (test-op min-2 [[7 3 1] [2 1 4]] [[1 2 3] [4 5 6]] [0 4 9])]
      (is (nd= v [[0 2 3] [0 4 6]]))
      (is (nd= da [[0 3 1] [0 0 4]]))
      (is (nd= db [9 1 0]))))
  (testing "min(matrix, row matrix)"
    (let [[v [da db]] (test-op min-2 [[7 3 1] [2 1 4]] [[1 2 3] [4 5 6]] [[0 4 9]])]
      (is (nd= v [[0 2 3] [0 4 6]]))
      (is (nd= da [[0 3 1] [0 0 4]]))
      (is (nd= db [[9 1 0]]))))
  (testing "min(matrix, matrix)"
    (let [[v [da db]] (test-op min-2 [[1 2] [3 4]] [[0 3] [2 6]] [[1 2] [3 0]])]
      (is (nd= v [[0 2] [2 0]]))
      (is (nd= da [[1 0] [3 0]]))
      (is (nd= db [[0 2] [0 4]])))))

(deftest abs-test
  (testing "abs(matrix)"
    (let [[v [da]] (test-op abs [[1 2] [1 2]] [[-1 2] [3 -4]])]
      (is (nd= v [[1 2] [3 4]]))
      (is (nd= da [[-1 2] [1 -2]])))))

(deftest negate-test
  (testing "negate(matrix)"
    (let [[v [da]] (test-op negate [[1 -2] [1 -2]] [[-1 2] [3 -4]])]
      (is (nd= v [[1 -2] [-3 4]]))
      (is (nd= da [[-1 2] [-1 2]])))))

(deftest exp-test
  (let [g (G (exp :a))]
    (testing "exp(1) == E"
      (is (nd= (evaluate g {:a 1}) Math/E)))
    (testing "exp(scalar)"
      (numerically-validated? g {:a 3}))
    (testing "exp(matrix)"
      (numerically-validated? g {:a [[-2 -1 0] [1 2 3]]} 0.01))))

(deftest log-test
  (let [g (G (log :a))]
    (testing "log(E) == 1"
      (is (nd= (evaluate g {:a Math/E}) 1)))
    (testing "log(1) == 0"
      (is (nd= (evaluate g {:a 1}) 0)))
    (testing "log(scalar)"
      (numerically-validated? g {:a 5})
      (numerically-validated? g {:a 0.5})
      (numerically-validated? g {:a 1}))
    (testing "log(matrix)"
      (numerically-validated? g {:a [[0.1 0.5 1]
                                     [2 3 5]]} 0.01))))

(deftest sum-along-test
  (testing "scalars - sum-along all dimensions"
    (let [[v [da]] (test-op sum-along 3 7)]
      (is (nd= v 7))
      (is (nd= da 3)))
    #_(let [[v da] (test-op-with-args sum-along 3 7 {:axis []})]
        (is (nd= v 7))
        (is (nd= da 3))))
  (testing "vectors - sum-along all dimensions"
    (let [[v [da]] (test-op sum-along 7 [1 2 3])]
      (is (nd= v 6))
      (is (nd= da [7 7 7]))))
  (testing "matrices - sum-along all dimensions"
    (let [[v [da]] (test-op sum-along 7 [[1 2 3] [4 5 6]])]
      (is (nd= v 21))
      (is (nd= da [[7 7 7] [7 7 7]]))))
  (testing "matrices - collapse"
    (let [[v da] (test-op-with-args sum-along [7 8 9] [[1 2 3] [4 5 6]] {:axis [0]})]
      (is (nd= v [5 7 9]))
      (is (nd= da [[7 8 9] [7 8 9]])))
    (let [[v da] (test-op-with-args sum-along [7 8] [[1 2 3] [4 5 6]] {:axis [1]})]
      (is (nd= v [6 15]))
      (is (nd= da [[7 7 7] [8 8 8]])))
    (let [[v da] (test-op-with-args sum-along 7 [[1 2 3] [4 5 6]] {:axis [0 1]})]
      (is (nd= v 21))
      (is (nd= da [[7 7 7] [7 7 7]]))))
  (testing "tensors - collapse"
    (let [[v da] (test-op-with-args sum-along [[10 11 12] [13 14 15]] [[[1 2 3] [4 5 6]]
                                                                       [[7 8 9] [0 1 2]]] {:axis [0]})]
      (is (nd= v [[8 10 12] [4 6 8]]))
      (is (nd= da [[[10 11 12] [13 14 15]] [[10 11 12] [13 14 15]]])))
    (let [[v da] (test-op-with-args sum-along [[10 11 12] [13 14 15]] [[[1 2 3] [4 5 6]]
                                                                       [[7 8 9] [0 1 2]]] {:axis [1]})]
      (is (nd= v [[5 7 9] [7 9 11]]))
      (is (nd= da [[[10 11 12] [10 11 12]] [[13 14 15] [13 14 15]]])))
    (let [[v da] (test-op-with-args sum-along [[10 11] [12 13]] [[[1 2 3] [4 5 6]]
                                                                 [[7 8 9] [0 1 2]]] {:axis [2]})]
      (is (nd= v [[6 15] [24 3]]))
      (is (nd= da [[[10 10 10] [11 11 11]] [[12 12 12] [13 13 13]]])))
    (let [[v da] (test-op-with-args sum-along [10 11 12] [[[1 2 3] [4 5 6]]
                                                          [[7 8 9] [0 1 2]]] {:axis [0 1]})]
      (is (nd= v [12 16 20]))
      (is (nd= da [[[10 11 12] [10 11 12]] [[10 11 12] [10 11 12]]])))
    (let [[v da] (test-op-with-args sum-along [10 11] [[[1 2 3] [4 5 6]]
                                                       [[7 8 9] [0 1 2]]] {:axis [0 2]})]
      (is (nd= v [30 18]))
      (is (nd= da [[[10 10 10] [11 11 11]] [[10 10 10] [11 11 11]]])))
    (let [[v da] (test-op-with-args sum-along [10 11] [[[1 2 3] [4 5 6]]
                                                       [[7 8 9] [0 1 2]]] {:axis [1 2]})]
      (is (nd= v [21 27]))
      (is (nd= da [[[10 10 10] [10 10 10]] [[11 11 11] [11 11 11]]])))
    (let [[v da] (test-op-with-args sum-along 10 [[[1 2 3] [4 5 6]]
                                                  [[7 8 9] [0 1 2]]] {:axis [0 1 2]})]
      (is (nd= v 48))
      (is (nd= da [[[10 10 10] [10 10 10]] [[10 10 10] [10 10 10]]]))))
  (testing "matrices - keep dimensions"
    (let [[v da] (test-op-with-args sum-along [7 8 9] [[1 2 3] [4 5 6]] {:axis [0]
                                                                         :collapse false})]
      (is (nd= v [[5 7 9]]))
      (is (nd= da [[7 8 9] [7 8 9]])))
    (let [[v da] (test-op-with-args sum-along [7 8] [[1 2 3] [4 5 6]] {:axis [1]
                                                                       :collapse false})]
      (is (nd= v [[6] [15]]))
      (is (nd= da [[7 7 7] [8 8 8]])))
    (let [[v da] (test-op-with-args sum-along 7 [[1 2 3] [4 5 6]] {:axis [0 1]
                                                                   :collapse false})]
      (is (nd= v [[21]]))
      (is (nd= da [[7 7 7] [7 7 7]]))))
  (testing "tensors - keep dimensions"
    (let [[v da] (test-op-with-args sum-along [[10 11 12] [13 14 15]] [[[1 2 3] [4 5 6]]
                                                                       [[7 8 9] [0 1 2]]] {:axis [0]
                                                                                           :collapse false})]
      (is (nd= v [[[8 10 12] [4 6 8]]]))
      (is (nd= da [[[10 11 12] [13 14 15]] [[10 11 12] [13 14 15]]])))
    (let [[v da] (test-op-with-args sum-along [[10 11 12] [13 14 15]] [[[1 2 3] [4 5 6]]
                                                                       [[7 8 9] [0 1 2]]] {:axis [1]
                                                                                           :collapse false})]
      (is (nd= v [[[5 7 9]] [[7 9 11]]]))
      (is (nd= da [[[10 11 12] [10 11 12]] [[13 14 15] [13 14 15]]])))
    (let [[v da] (test-op-with-args sum-along [[10 11] [12 13]] [[[1 2 3] [4 5 6]]
                                                                 [[7 8 9] [0 1 2]]] {:axis [2]
                                                                                     :collapse false})]
      (is (nd= v [[[6] [15]] [[24] [3]]]))
      (is (nd= da [[[10 10 10] [11 11 11]] [[12 12 12] [13 13 13]]])))
    (let [[v da] (test-op-with-args sum-along [10 11 12] [[[1 2 3] [4 5 6]]
                                                          [[7 8 9] [0 1 2]]] {:axis [0 1]
                                                                              :collapse false})]
      (is (nd= v [[[12 16 20]]]))
      (is (nd= da [[[10 11 12] [10 11 12]] [[10 11 12] [10 11 12]]])))
    (let [[v da] (test-op-with-args sum-along [10 11] [[[1 2 3] [4 5 6]]
                                                       [[7 8 9] [0 1 2]]] {:axis [0 2]
                                                                           :collapse false})]
      (is (nd= v [[[30] [18]]]))
      (is (nd= da [[[10 10 10] [11 11 11]] [[10 10 10] [11 11 11]]])))
    (let [[v da] (test-op-with-args sum-along [10 11] [[[1 2 3] [4 5 6]]
                                                       [[7 8 9] [0 1 2]]] {:axis [1 2]
                                                                           :collapse false})]
      (is (nd= v [[[21]] [[27]]]))
      (is (nd= da [[[10 10 10] [10 10 10]] [[11 11 11] [11 11 11]]])))
    (let [[v da] (test-op-with-args sum-along 10 [[[1 2 3] [4 5 6]]
                                                  [[7 8 9] [0 1 2]]] {:axis [0 1 2]
                                                                      :collapse false})]
      (is (nd= v [[[48]]]))
      (is (nd= da [[[10 10 10] [10 10 10]] [[10 10 10] [10 10 10]]])))))

(deftest max-along-test
  (testing "scalars - max along all dimensions"
    (let [[v [da]] (test-op max-along 3 7)]
      (is (nd= v 7))
      (is (nd= da 3)))
    #_(let [[v da] (test-op-with-args max-along 3 7 {:axis []})]
        (is (nd= v 7))
        (is (nd= da 3))))
  (testing "vectors - max along all dimensions"
    (let [[v [da]] (test-op max-along 7 [1 3 2])]
      (is (nd= v 3))
      (is (nd= da [0 7 0]))))
  (testing "matrices - max along all dimensions"
    (let [[v [da]] (test-op max-along 7 [[1 3 2] [6 5 6]])]
      (is (nd= v 6))
      (is (nd= da [[0 0 0] [7 0 7]]))))
  (testing "matrices - collapse"
    (let [[v da] (test-op-with-args max-along [7 8] [[4 2] [1 5]] {:axis [0]})]
      (is (nd= v [4 5]))
      (is (nd= da [[7 0] [0 8]])))
    (let [[v da] (test-op-with-args max-along [7 8] [[6 3] [1 5]] {:axis [1]})]
      (is (nd= v [6 5]))
      (is (nd= da [[7 0] [0 8]])))
    (let [[v da] (test-op-with-args max-along 7 [[6 3 2] [6 5 4]] {:axis [0 1]})]
      (is (nd= v 6))
      (is (nd= da [[7 0 0] [7 0 0]]))))
  (testing "tensors - collapse"
    (let [[v da] (test-op-with-args max-along [[10 11 12] [13 14 15]] [[[1 8 3] [4 1 6]]
                                                                       [[7 2 9] [4 5 2]]] {:axis [0]})]
      (is (nd= v [[7 8 9] [4 5 6]]))
      (is (nd= da [[[0 11 0] [13 0 15]] [[10 0 12] [13 14 0]]])))
    (let [[v da] (test-op-with-args max-along [[10 11 12] [13 14 15]] [[[1 8 3] [4 1 6]]
                                                                       [[7 2 9] [4 5 2]]] {:axis [1]})]
      (is (nd= v [[4 8 6] [7 5 9]]))
      (is (nd= da [[[0 11 0] [10 0 12]] [[13 0 15] [0 14 0]]])))
    (let [[v da] (test-op-with-args max-along [[10 11] [12 13]] [[[1 8 3] [4 1 6]]
                                                                 [[7 2 9] [4 5 2]]] {:axis [2]})]
      (is (nd= v [[8 6] [9 5]]))
      (is (nd= da [[[0 10 0] [0 0 11]] [[0 0 12] [0 13 0]]])))
    (let [[v da] (test-op-with-args max-along [10 11 12] [[[1 8 3] [4 1 6]]
                                                          [[7 2 9] [4 5 2]]] {:axis [0 1]})]
      (is (nd= v [7 8 9]))
      (is (nd= da [[[0 11 0] [0 0 0]] [[10 0 12] [0 0 0]]])))
    (let [[v da] (test-op-with-args max-along [10 11] [[[1 8 3] [4 1 6]]
                                                       [[7 2 9] [4 5 2]]] {:axis [0 2]})]
      (is (nd= v [9 6]))
      (is (nd= da [[[0 0 0] [0 0 11]] [[0 0 10] [0 0 0]]])))
    (let [[v da] (test-op-with-args max-along [10 11] [[[1 2 3] [4 5 6]]
                                                       [[7 8 9] [0 1 2]]] {:axis [1 2]})]
      (is (nd= v [6 9]))
      (is (nd= da [[[0 0 0] [0 0 10]] [[0 0 11] [0 0 0]]])))
    (let [[v da] (test-op-with-args max-along 10 [[[1 8 3] [4 1 6]]
                                                  [[7 2 9] [4 5 2]]] {:axis [0 1 2]})]
      (is (nd= v 9))
      (is (nd= da [[[0 0 0] [0 0 0]] [[0 0 10] [0 0 0]]]))))
  (testing "matrices - keep dimensions"
    (let [[v da] (test-op-with-args max-along [7 8 9] [[6 3 2] [6 5 4]] {:axis [0]
                                                                         :collapse false})]
      (is (nd= v [[6 5 4]]))
      (is (nd= da [[7 0 0] [7 8 9]])))
    (let [[v da] (test-op-with-args max-along [7 8] [[6 5 2] [1 5 4]] {:axis [1]
                                                                       :collapse false})]
      (is (nd= v [[6] [5]]))
      (is (nd= da [[7 0 0] [0 8 0]])))
    (let [[v da] (test-op-with-args max-along 7 [[1 3 2] [6 5 4]] {:axis [0 1]
                                                                   :collapse false})]
      (is (nd= v [[6]]))
      (is (nd= da [[0 0 0] [7 0 0]]))))
  (testing "tensors - keep dimensions"
    (let [[v da] (test-op-with-args max-along [[[10 11 12] [13 14 15]]] [[[1 8 3] [4 1 6]]
                                                                         [[7 2 9] [4 5 2]]] {:axis [0]
                                                                                             :collapse false})]
      (is (nd= v [[[7 8 9] [4 5 6]]]))
      (is (nd= da [[[0 11 0] [13 0 15]] [[10 0 12] [13 14 0]]])))
    (let [[v da] (test-op-with-args max-along [[[10 11 12]] [[13 14 15]]] [[[1 8 3] [4 1 6]]
                                                                           [[7 2 9] [4 5 2]]] {:axis [1]
                                                                                               :collapse false})]
      (is (nd= v [[[4 8 6]] [[7 5 9]]]))
      (is (nd= da [[[0 11 0] [10 0 12]] [[13 0 15] [0 14 0]]])))
    (let [[v da] (test-op-with-args max-along [[[10] [11]] [[12] [13]]] [[[1 8 3] [4 1 6]]
                                                                         [[7 2 9] [4 5 2]]] {:axis [2]
                                                                                             :collapse false})]
      (is (nd= v [[[8] [6]] [[9] [5]]]))
      (is (nd= da [[[0 10 0] [0 0 11]] [[0 0 12] [0 13 0]]])))
    (let [[v da] (test-op-with-args max-along [[[10 11 12]]] [[[1 8 3] [4 1 6]]
                                                              [[7 2 9] [4 5 2]]] {:axis [0 1]
                                                                                  :collapse false})]
      (is (nd= v [[[7 8 9]]]))
      (is (nd= da [[[0 11 0] [0 0 0]] [[10 0 12] [0 0 0]]])))
    (let [[v da] (test-op-with-args max-along [[[10] [11]]] [[[1 8 3] [4 1 6]]
                                                             [[7 2 9] [4 5 2]]] {:axis [0 2]
                                                                                 :collapse false})]
      (is (nd= v [[[9] [6]]]))
      (is (nd= da [[[0 0 0] [0 0 11]] [[0 0 10] [0 0 0]]])))
    (let [[v da] (test-op-with-args max-along [[[10] [11]]] [[[1 2 3] [4 5 6]]
                                                             [[7 8 9] [0 1 2]]] {:axis [1 2]
                                                                                 :collapse false})]
      (is (nd= v (m/array [[[6]] [[9]]])))
      (is (nd= da [[[0 0 0] [0 0 10]] [[0 0 11] [0 0 0]]])))
    (let [[v da] (test-op-with-args max-along [[[10]]] [[[1 8 3] [4 1 6]]
                                                        [[7 2 9] [4 5 2]]] {:axis [0 1 2]
                                                                            :collapse false})]
      (is (nd= v [[[9]]]))
      (is (nd= da [[[0 0 0] [0 0 0]] [[0 0 10] [0 0 0]]])))))

(deftest min-along-test
  (testing "scalars - min along all dimensions"
    (let [[v [da]] (test-op min-along 3 7)]
      (is (nd= v 7))
      (is (nd= da 3)))
    #_(let [[v da] (test-op-with-args min-along 3 7 {:axis []})]
        (is (nd= v 7))
        (is (nd= da 3))))
  (testing "vectors - min along all dimensions"
    (let [[v [da]] (test-op min-along 7 [1 3 2])]
      (is (nd= v 1))
      (is (nd= da [7 0 0]))))
  (testing "matrices - min along all dimensions"
    (let [[v [da]] (test-op min-along 7 [[2 3 1] [1 5 6]])]
      (is (nd= v 1))
      (is (nd= da [[0 0 7] [7 0 0]]))))
  (testing "matrices - collapse"
    (let [[v da] (test-op-with-args min-along [7 8] [[4 2] [1 5]] {:axis [0]})]
      (is (nd= v [1 2]))
      (is (nd= da [[0 8] [7 0]])))
    (let [[v da] (test-op-with-args min-along [7 8] [[6 3] [1 5]] {:axis [1]})]
      (is (nd= v [3 1]))
      (is (nd= da [[0 7] [8 0]])))
    (let [[v da] (test-op-with-args min-along 7 [[6 2 2] [6 5 4]] {:axis [0 1]})]
      (is (nd= v 2))
      (is (nd= da [[0 7 7] [0 0 0]]))))
  (testing "tensors - collapse"
    (let [[v da] (test-op-with-args min-along [[10 11] [13 14]] [[[1 8] [4 1]]
                                                                 [[7 2] [4 5]]] {:axis [0]})]
      (is (nd= v [[1 2] [4 1]]))
      (is (nd= da [[[10 0] [13 14]] [[0 11] [13 0]]])))
    (let [[v da] (test-op-with-args min-along [[10 11] [13 14]] [[[1 8] [4 1]]
                                                                 [[7 2] [4 5]]] {:axis [1]})]
      (is (nd= v [[1 1] [4 2]]))
      (is (nd= da [[[10 0] [0 11]] [[0 14] [13 0]]])))
    (let [[v da] (test-op-with-args min-along [[10 11] [12 13]] [[[1 8] [4 1]]
                                                                 [[7 2] [4 5]]] {:axis [2]})]
      (is (nd= v [[1 1] [2 4]]))
      (is (nd= da [[[10 0] [0 11]] [[0 12] [13 0]]])))
    (let [[v da] (test-op-with-args min-along [10 11] [[[1 8] [4 1]]
                                                       [[7 2] [4 5]]] {:axis [0 1]})]
      (is (nd= v [1 1]))
      (is (nd= da [[[10 0] [0 11]] [[0 0] [0 0]]])))
    (let [[v da] (test-op-with-args min-along [10 11] [[[1 8] [4 1]]
                                                       [[7 2] [4 5]]] {:axis [0 2]})]
      (is (nd= v [1 1]))
      (is (nd= da [[[10 0] [0 11]] [[0 0] [0 0]]])))
    (let [[v da] (test-op-with-args min-along [10 11] [[[1 2] [4 5]]
                                                       [[7 8] [0 1]]] {:axis [1 2]})]
      (is (nd= v [1 0]))
      (is (nd= da [[[10 0] [0 0]] [[0 0] [11 0]]])))
    (let [[v da] (test-op-with-args min-along 10 [[[1 8] [4 1]]
                                                  [[7 2] [4 5]]] {:axis [0 1 2]})]
      (is (nd= v 1))
      (is (nd= da [[[10 0] [0 10]] [[0 0] [0 0]]]))))
  (testing "matrices - keep dimensions"
    (let [[v da] (test-op-with-args min-along [[7 8]] [[4 2] [1 5]] {:axis [0]
                                                                     :collapse false})]
      (is (nd= v [[1 2]]))
      (is (nd= da [[0 8] [7 0]])))
    (let [[v da] (test-op-with-args min-along [[7] [8]] [[6 3] [1 5]] {:axis [1]
                                                                       :collapse false})]
      (is (nd= v [[3] [1]]))
      (is (nd= da [[0 7] [8 0]])))
    (let [[v da] (test-op-with-args min-along [[7]] [[6 2 2] [6 5 4]] {:axis [0 1]
                                                                       :collapse false})]
      (is (nd= v [[2]]))
      (is (nd= da [[0 7 7] [0 0 0]]))))
  (testing "tensors - keep dimensions"
    (let [[v da] (test-op-with-args min-along [[[10 11] [13 14]]] [[[1 8] [4 1]]
                                                                   [[7 2] [4 5]]] {:axis [0]
                                                                                   :collapse false})]
      (is (nd= v [[[1 2] [4 1]]]))
      (is (nd= da [[[10 0] [13 14]] [[0 11] [13 0]]])))
    (let [[v da] (test-op-with-args min-along [[[10 11]] [[13 14]]] [[[1 8] [4 1]]
                                                                     [[7 2] [4 5]]] {:axis [1]
                                                                                     :collapse false})]
      (is (nd= v [[[1 1]] [[4 2]]]))
      (is (nd= da [[[10 0] [0 11]] [[0 14] [13 0]]])))
    (let [[v da] (test-op-with-args min-along [[[10] [11]] [[12] [13]]] [[[1 8] [4 1]]
                                                                         [[7 2] [4 5]]] {:axis [2]
                                                                                         :collapse false})]
      (is (nd= v [[[1] [1]] [[2] [4]]]))
      (is (nd= da [[[10 0] [0 11]] [[0 12] [13 0]]])))
    (let [[v da] (test-op-with-args min-along [[[10 11]]] [[[1 8] [4 1]]
                                                           [[7 2] [4 5]]] {:axis [0 1]
                                                                           :collapse false})]
      (is (nd= v [[[1 1]]]))
      (is (nd= da [[[10 0] [0 11]] [[0 0] [0 0]]])))
    (let [[v da] (test-op-with-args min-along [[[10] [11]]] [[[1 8] [4 1]]
                                                             [[7 2] [4 5]]] {:axis [0 2]
                                                                             :collapse false})]
      (is (nd= v [[[1] [1]]]))
      (is (nd= da [[[10 0] [0 11]] [[0 0] [0 0]]])))
    (let [[v da] (test-op-with-args min-along [[[10]] [[11]]] [[[1 2] [4 5]]
                                                               [[7 8] [0 1]]] {:axis [1 2]
                                                                               :collapse false})]
      (is (nd= v [[[1]] [[0]]]))
      (is (nd= da [[[10 0] [0 0]] [[0 0] [11 0]]])))
    (let [[v da] (test-op-with-args min-along [[[10]]] [[[1 8] [4 1]]
                                                        [[7 2] [4 5]]] {:axis [0 1 2]
                                                                        :collapse false})]
      (is (nd= v [[[1]]]))
      (is (nd= da [[[10 0] [0 10]] [[0 0] [0 0]]])))))

(deftest max-mask-test
  (testing "is non-backpropagating"
    #_(let [[_ grads] (test-op max-mask 3 7 [])]
        (= grads [nil nil])))
  ; TODO: Support this op for scalars with nd4j or else remove these assertions.
  #_(testing "scalars"
      (evaluates-to? (max-mask 7 []) 1))
  (testing "vectors"
    (evaluates-to? (max-mask [2 1 3 1] [0]) [0 0 1 0]))
  (testing "matrices"
    (evaluates-to? (max-mask [[2 3] [1 4]] 0) [[1 0] [0 1]])
    (evaluates-to? (max-mask [[4 3] [1 4]] [0]) [[1 0] [0 1]])
    (evaluates-to? (max-mask [[2 3] [1 4]] 1) [[0 1] [0 1]])
    (evaluates-to? (max-mask [[2 3] [1 4]] [0 1]) [[0 0] [0 1]]))
  (testing "tensors"
    (evaluates-to? (max-mask [[[2 1 6] [5 0 7]]
                              [[1 0 8] [3 2 4]]] [0]) [[[1 1 0] [1 0 1]]
                                                       [[0 0 1] [0 1 0]]])
    (evaluates-to? (max-mask [[[2 1 6] [5 0 7]]
                              [[1 0 8] [3 2 4]]] [1]) [[[0 1 0] [1 0 1]]
                                                       [[0 0 1] [1 1 0]]])
    (evaluates-to? (max-mask [[[2 1 6] [5 0 7]]
                              [[1 0 8] [3 2 4]]] [2]) [[[0 0 1] [0 0 1]]
                                                       [[0 0 1] [0 0 1]]])
    (evaluates-to? (max-mask [[[2 1 6] [5 0 7]]
                              [[1 0 8] [3 2 4]]] [0 1]) [[[0 0 0] [1 0 0]]
                                                         [[0 0 1] [0 1 0]]])
    (evaluates-to? (max-mask [[[2 1 6] [5 0 7]]
                              [[1 0 8] [3 2 4]]] [0 2]) [[[0 0 0] [0 0 1]]
                                                         [[0 0 1] [0 0 0]]])
    (evaluates-to? (max-mask [[[2 1 6] [5 0 7]]
                              [[1 0 8] [3 2 4]]] [1 2]) [[[0 0 0] [0 0 1]]
                                                         [[0 0 1] [0 0 0]]])
    (evaluates-to? (max-mask [[[2 1 6] [5 0 7]]
                              [[1 0 8] [3 2 4]]] [0 1 2]) [[[0 0 0] [0 0 0]]
                                                           [[0 0 1] [0 0 0]]])))

(deftest min-mask-test
  #_(testing "is non-backpropagating"
      (let [[_ grads] (test-op min-mask 3 7 [])]
        (= grads [nil nil])))
  #_(testing "scalars"
      (evaluates-to? (min-mask 7 []) 1))
  (testing "vectors"
    (evaluates-to? (min-mask [2 1 3 1] [0]) [0 1 0 1]))
  (testing "matrices"
    (evaluates-to? (min-mask [[2 3] [1 4]] 0) [[0 1] [1 0]])
    (evaluates-to? (min-mask [[1 3] [1 4]] [0]) [[1 1] [1 0]])
    (evaluates-to? (min-mask [[2 3] [1 4]] 1) [[1 0] [1 0]])
    (evaluates-to? (min-mask [[2 3] [1 4]] [0 1]) [[0 0] [1 0]]))
  (testing "tensors"
    (evaluates-to? (min-mask [[[2 1 6] [5 0 7]]
                              [[1 0 8] [3 2 4]]] [0]) [[[0 0 1] [0 1 0]]
                                                       [[1 1 0] [1 0 1]]])
    (evaluates-to? (min-mask [[[2 1 6] [5 0 7]]
                              [[1 0 8] [3 2 4]]] [1]) [[[1 0 1] [0 1 0]]
                                                       [[1 1 0] [0 0 1]]])
    (evaluates-to? (min-mask [[[2 1 6] [5 0 7]]
                              [[1 0 8] [3 2 4]]] [2]) [[[0 1 0] [0 1 0]]
                                                       [[0 1 0] [0 1 0]]])
    (evaluates-to? (min-mask [[[2 1 6] [5 0 7]]
                              [[1 0 8] [3 2 4]]] [0 1]) [[[0 0 0] [0 1 0]]
                                                         [[1 1 0] [0 0 1]]])
    (evaluates-to? (min-mask [[[2 1 6] [5 0 7]]
                              [[1 0 8] [3 2 4]]] [0 2]) [[[0 0 0] [0 1 0]]
                                                         [[0 1 0] [0 0 0]]])
    (evaluates-to? (min-mask [[[2 1 6] [5 0 7]]
                              [[1 0 8] [3 2 4]]] [1 2]) [[[0 0 0] [0 1 0]]
                                                         [[0 1 0] [0 0 0]]])
    (evaluates-to? (min-mask [[[2 1 6] [5 0 7]]
                              [[1 0 8] [3 2 4]]] [0 1 2]) [[[0 0 0] [0 1 0]]
                                                           [[0 1 0] [0 0 0]]])))

(deftest add-test
  (testing "0 operands"
    (let [[v grads] (test-op add 3)]
      (is (nd= v 0))
      (is (= grads []))))
  (testing "1 operand"
    (let [[v [da]] (test-op add [[5 6] [7 8]] [[1 2] [3 4]])]
      (is (nd= v [[1 2] [3 4]]))
      (is (nd= da [[5 6] [7 8]]))))
  (testing "2 operands"
    (let [[v [a b]] (test-op add [[5 6] [7 8]] [[1 2] [3 4]] [[9 10] [11 12]])]
      (is (nd= v [[10 12] [14 16]]))
      (is (nd= a [[5 6] [7 8]]))
      (is (nd= b [[5 6] [7 8]]))))
  (testing "3 operands"
    (let [[v [a b c]] (test-op add [[5 6] [7 8]]
                               [[1 2] [3 4]] [[9 10] [11 12]] [[-4 2] [7 -6]])]
      (is (nd= v [[6 14] [21 10]]))
      (is (nd= a [[5 6] [7 8]]))
      (is (nd= b [[5 6] [7 8]]))
      (is (nd= c [[5 6] [7 8]])))))

(deftest sub-test
  (testing "0 operands"
    (let [[v grads] (test-op sub 3)]
      (is (nd= v 0))
      (is (= grads []))))
  (testing "1 operand"
    (let [[v [da]] (test-op sub [[5 6] [7 8]] [[1 2] [3 4]])]
      (is (nd= v [[-1 -2] [-3 -4]]))
      (is (nd= da [[-5 -6] [-7 -8]]))))
  (testing "2 operands"
    (let [[v [da db]] (test-op sub [[5 6] [7 8]] [[9 10] [11 12]] [[1 4] [14 -3]])]
      (is (nd= v [[8 6] [-3 15]]))
      (is (nd= da [[5 6] [7 8]]))
      (is (nd= db [[-5 -6] [-7 -8]]))))
  (testing "3 operands"
    (let [[v [da db dc]] (test-op sub [[5 6] [7 8]]
                                  [[9 10] [11 12]] [[0 5] [10 3]] [[-4 2] [7 -6]])]
      (is (nd= v [[13 3] [-6 15]]))
      (is (nd= da [[5 6] [7 8]]))
      (is (nd= db [[-5 -6] [-7 -8]]))
      (is (nd= dc [[-5 -6] [-7 -8]])))))

(deftest mul-test
  (testing "0 operands"
    (let [[v grads] (test-op mul 3)]
      (is (nd= v 1))
      (is (= grads []))))
  (testing "1 operand"
    (let [[v [da]] (test-op mul [[5 6] [7 8]] [[1 2] [3 4]])]
      (is (nd= v [[1 2] [3 4]]))
      (is (nd= da [[5 6] [7 8]]))))
  (testing "2 operands"
    (let [[v [da db]] (test-op mul [[1 2] [3 4]] [[3 2] [1 0]] [[1 4] [14 -3]])]
      (is (nd= v [[3 8] [14 0]]))
      (is (nd= da [[1 8] [42 -12]]))
      (is (nd= db [[3 4] [3 0]]))))
  (testing "3 operands"
    (let [[v [da db dc]] (test-op mul [[1 2] [3 4]]
                                  [[3 2] [1 0]] [[1 4] [14 -3]] [[7 6] [5 4]])]
      (is (about= v [[21 48] [70 0]]))
      (is (nd= da [[7 48] [210 -48]]))
      (is (nd= db [[21 24] [15 0]]))
      (is (nd= dc [[3 16] [42 0]])))))

(deftest div-test
  (testing "1 operand"
    (let [g (G (/ :a))]
      (is (about= (evaluate g {:a 2}) 0.5))
      (numerically-validated? g {:a 2})
      (is (about= (evaluate g {:a [[10 4] [0.25 0.1]]}) [[0.1 0.25] [4 10]]))
      (numerically-validated? g {:a [[10 4] [0.25 0.1]]} 0.1)))
  (testing "2 operands"
    (let [g (G (/ :a :b))
          input {:a [[[1 2] [3 4]] [[5 6] [7 8]]]
                 :b [[[10 4] [3 0.5]] [[1/3 18] [3 80]]]}]
      (is (about= (evaluate g input) [[[0.1 0.5] [1 8]] [[15 1/3] [7/3 0.1]]]))))
  (testing "3 operands"
    (testing "matrix / matrix"
      (let [g (G (/ :a :b :c))
            input {:a [[1 2] [7 8]]
                   :b [[5 6] [3 4]]
                   :c [0.1 1/3]}]
        (is (about= (evaluate g input) [[2 1] [70/3 6]]))))))

(deftest max-test
  (testing "1 operand"
    (let [[v [da]] (test-op r/max [[5 6] [7 8]] [[1 2] [3 4]])]
      (is (nd= v [[1 2] [3 4]]))
      (is (nd= da [[5 6] [7 8]]))))
  (testing "2 operands"
    (let [[v [da db]] (test-op r/max [[1 2] [3 4]] [[7 2] [6 -3]] [[1 4] [6 2]])]
      (is (nd= v [[7 4] [6 2]]))
      (is (nd= da [[1 0] [3 0]]))
      (is (nd= db [[0 2] [3 4]]))))
  (testing "3 operands"
    (let [[v [da db dc]] (test-op r/max [[1 2] [3 4]]
                                  [[7 2] [6 -3]] [[1 4] [6 2]] [[9 0] [-5 3]])]
      (is (nd= v [[9 4] [6 3]]))
      (is (nd= da [[0 0] [3 0]]))
      (is (nd= db [[0 2] [3 0]]))
      (is (nd= dc [[1 0] [0 4]])))))

(deftest min-test
  (testing "1 operand"
    (let [[v [da]] (test-op r/min [[5 6] [7 8]] [[1 2] [3 4]])]
      (is (nd= v [[1 2] [3 4]]))
      (is (nd= da [[5 6] [7 8]]))))
  (testing "2 operands"
    (let [[v [da db]] (test-op r/min [[1 2] [3 4]] [[7 2] [6 -3]] [[1 4] [6 2]])]
      (is (nd= v [[1 2] [6 -3]]))
      (is (nd= da [[0 2] [3 4]]))
      (is (nd= db [[1 0] [3 0]]))))
  (testing "3 operands"
    (let [[v [da db dc]] (test-op r/min [[1 2] [3 4]]
                                  [[7 2] [6 -3]] [[1 4] [6 2]] [[9 0] [-5 3]])]
      (is (nd= v [[1 0] [-5 -3]]))
      (is (nd= da [[0 0] [0 4]]))
      (is (nd= db [[1 0] [0 0]]))
      (is (nd= dc [[0 2] [3 0]])))))

(deftest transpose-test
  (testing "scalar"
    (let [[v [da]] (test-op transpose 7 3)]
      (is (nd= v 3))
      (is (nd= da 7))))
  (testing "vector"
    (let [[v [da]] (test-op transpose [4 5 6] [1 2 3])]
      (is (nd= v [1 2 3]))
      (is (nd= da [4 5 6]))))
  (testing "matrix"
    (let [[v [da]] (test-op transpose [[10 40] [20 50] [30 60]] [[1 2 3] [4 5 6]])]
      (is (nd= v [[1 4] [2 5] [3 6]]))
      (is (nd= da [[10 20 30] [40 50 60]]))))
  (testing "tensor"
    (let [[v [da]] (test-op transpose [[[10 70] [40 100]] [[20 80] [50 110]] [[30 90] [60 120]]]
                            [[[1 2 3] [4 5 6]] [[7 8 9] [10 11 12]]])]
      (is (nd= v [[[1 7] [4 10]] [[2 8] [5 11]] [[3 9] [6 12]]]))
      (is (nd= da [[[10 20 30] [40 50 60]] [[70 80 90] [100 110 120]]])))))

(deftest size-axes-test
  #_(testing "scalar"
      (let [[v [dnd daxis]] (test-op size-axes 7 3 [])]
        (is (= v 1))
        (is (nil? dnd))
        (is (nil? daxis)))
      (is (thrown? Exception (test-op size-axes 7 3 [1]))))
  (testing "vector"
    (is (nd= 3 (evaluate (size-axes [1 2 3] [0]))))
    (is (thrown? Exception (test-op size-axes 7 [1 2 3] [1]))))
  (testing "matrix"
    (is (nd= 2 (evaluate (size-axes [[1 2 3] [4 5 6]] [0]))))
    (is (nd= 3 (evaluate (size-axes [[1 2 3] [4 5 6]] [1]))))
    (is (thrown? Exception (test-op size-axes 7 [[1 2 3] [4 5 6]] [2]))))
  (testing "tensor"
    (let [g (G (size-axes [[[1 2 3]] [[4 5 6]]] (input :axis false)))]
      (is (nd= (evaluate g {:axis [0]}) 2))
      (is (nd= (evaluate g {:axis [1]}) 1))
      (is (nd= (evaluate g {:axis [2]}) 3))
      (is (nd= (evaluate g {:axis [0 1]}) 2))
      (is (nd= (evaluate g {:axis [0 2]}) 6))
      (is (nd= (evaluate g {:axis [1 2]}) 3))
      (is (nd= (evaluate g {:axis [0 1 2]}) 6))
      (is (thrown? Exception (evaluate g {:axis [3]}))))))

(deftest size-test
  (is (nd= (evaluate (G (size 3 {:axis 0}))) 1))
  (is (nd= (evaluate (G (size [1 2 3] {:axis 0}))) 3))
  (is (thrown? Exception (evaluate (G (size [1 2 3] {:axis 1}))) 3))
  (is (nd= (evaluate (G (size [[1 2 3] [4 5 6]] {:axis 0}))) 2))
  (is (nd= (evaluate (G (size [[1 2 3] [4 5 6]] {:axis 1}))) 3))
  (is (nd= (evaluate (G (size [[1 2 3] [4 5 6]] {:axis [0 1]}))) 6))
  (is (nd= (evaluate (G (size [[1 2 3] [4 5 6]]))) 6))
  (is (nd= (evaluate (G (size [[[1 2 3]] [[1 2 3]]]))) 6))
  (is (nd= (evaluate (G (size [[[1 2 3]] [[1 2 3]]] {:axis 0}))) 2))
  (is (nd= (evaluate (G (size [[[1 2 3]] [[1 2 3]]] {:axis 1}))) 1))
  (is (nd= (evaluate (G (size [[[1 2 3]] [[1 2 3]]] {:axis 2}))) 3))
  (is (nd= (evaluate (G (size [[[1 2 3]] [[1 2 3]]] {:axis [0 1]}))) 2))
  (is (nd= (evaluate (G (size [[[1 2 3]] [[1 2 3]]] {:axis [1 2]}))) 3))
  (is (nd= (evaluate (G (size [[[1 2 3]] [[1 2 3]]] {:axis [0 2]}))) 6)))

(deftest shape-test
  (testing "backpropagates nil"
    (let [[_ [grad]] (test-op shape [3 7] [[0 0] [0 0]])]
      (is (= grad nil))))
  (testing "computes shapes correctly"
    #_(evaluates-to? (shape 3) [])
    (evaluates-to? (shape [1 2 3]) [3])
    (evaluates-to? (shape [[1 2 3] [4 5 6]]) [2 3])
    (evaluates-to? (shape [[[1 2 3] [4 5 6]] [[4 5 6] [7 8 9]]]) [2 2 3])))

(deftest reshape-test
  (testing "reshapes scalars"
    #_(let [[v [grad]] (test-op reshape 7 3 [])]
        (is (nd= v 3))
        (is (nd= grad 7)))
    (let [[v [grad]] (test-op reshape [7] 3 [1])]
      (is (nd= v [3]))
      (is (nd= grad 7)))
    (let [[v [grad]] (test-op reshape [[7]] 3 [1 1])]
      (is (nd= v [[3]]))
      (is (nd= grad 7)))
    (let [[v [grad]] (test-op reshape [[[7]]] 3 [1 1 1])]
      (is (nd= v [[[3]]]))
      (is (nd= grad 7))))
  (testing "reshapes vectors"
    (let [[v [grad]] (test-op reshape [[5 6] [7 8]] [1 2 3 4] [2 2])]
      (is (nd= v [[1 2] [3 4]]))
      (is (nd= grad [5 6 7 8])))
    (let [[v [grad]] (test-op reshape [5 6 7 8] [1 2 3 4] [4])]
      (is (nd= v [1 2 3 4]))
      (is (nd= grad [5 6 7 8]))))
  (testing "reshapes matrices"
    (let [[v [grad]] (test-op reshape [4 5 6] [[1 2 3]] [3])]
      (is (nd= v [1 2 3]))
      (is (nd= grad [[4 5 6]])))
    (let [[v [grad]] (test-op reshape [4 5 6] [[1] [2] [3]] [3])]
      (is (nd= v [1 2 3]))
      (is (nd= grad [[4] [5] [6]])))
    (let [[v [grad]] (test-op reshape [[5 6] [7 8]] [[1 2] [3 4]] [2 2])]
      (is (nd= v [[1 2] [3 4]]))
      (is (nd= grad [[5 6] [7 8]])))
    (let [[v [grad]] (test-op reshape [[[5 6] [7 8]]] [[1 2] [3 4]] [1 2 2])]
      (is (nd= v [[[1 2] [3 4]]]))
      (is (nd= grad [[5 6] [7 8]])))
    (let [[v [grad]] (test-op reshape [[[5 6]] [[7 8]]] [[1 2] [3 4]] [2 1 2])]
      (is (nd= v [[[1 2]] [[3 4]]]))
      (is (nd= grad [[5 6] [7 8]])))
    (let [[v [grad]] (test-op reshape [[[5] [6]] [[[7] [8]]]] [[1 2] [3 4]] [2 2 1])]
      (is (nd= v [[[1] [2]] [[3] [4]]]))
      (is (nd= grad [[5 6] [7 8]])))
    (let [[v [grad]] (test-op reshape [[7 8 9] [10 11 12]] [[1 2 3] [4 5 6]] [3 2])]
      (is (nd= v [[1 2] [3 4] [5 6]]))
      (is (nd= grad [[7 8 9] [10 11 12]]))))
  (testing "reshapes tensors"
    (let [[v [grad]] (test-op reshape [4 5 6] [[[1 2 3]]] [3])]
      (is (nd= v [1 2 3]))
      (is (nd= grad [[[4 5 6]]])))
    (let [[v [grad]] (test-op reshape [[4 5] [6 7]] [[[1] [2] [3] [4]]] [2 2])]
      (is (nd= v [[1 2] [3 4]]))
      (is (nd= grad [[[4] [5] [6] [7]]])))
    (let [[v [grad]] (test-op reshape [[[[[4 5]] [[6 7]]]]] [[[1] [2] [3] [4]]] [1 1 2 1 2])]
      (is (nd= v [[[[[1 2]] [[3 4]]]]]))
      (is (nd= grad [[[4] [5] [6] [7]]]))))
  (testing "throws an error for incompatible shapes"
    (is (thrown? Exception (evaluate (reshape 3 [2]))))
    (is (thrown? Exception (evaluate (reshape [[1 2 3]] [1 4]))))
    (is (thrown? Exception (evaluate (reshape [[1 2 3]] [4 1]))))
    (is (thrown? Exception (evaluate (reshape [[1 2 3]] [1 3 2]))))))

(deftest join-2-along-test
  (testing "joins vectors"
    (let [[v [_ da db]] (test-op join-2-along [3 4] 0 [1] [2])]
      (is (nd= v [1 2]))
      (is (nd= da [3]))
      (is (nd= db [4])))
    (let [[v [_ da db]] (test-op join-2-along [4 5 6] 0 [1 2] [3])]
      (is (nd= v [1 2 3]))
      (is (nd= da [4 5]))
      (is (nd= db [6])))
    (let [[v [_ da db]] (test-op join-2-along [4 5 6] 0 [1] [2 3])]
      (is (nd= v [1 2 3]))
      (is (nd= da [4]))
      (is (nd= db [5 6]))))
  (testing "joins matrices"
    (let [[v [_ da db]] (test-op join-2-along [[9 10] [11 12] [13 14] [15 16]] 0 [[1 2] [3 4]] [[5 6] [7 8]])]
      (is (nd= v [[1 2] [3 4] [5 6] [7 8]]))
      (is (nd= da [[9 10] [11 12]]))
      (is (nd= db [[13 14] [15 16]])))
    (let [[v [_ da db]] (test-op join-2-along [[9 10 13 14] [11 12 15 16]] 1 [[1 2] [3 4]] [[5 6] [7 8]])]
      (is (nd= v [[1 2 5 6] [3 4 7 8]]))
      (is (nd= da [[9 10] [11 12]]))
      (is (nd= db [[13 14] [15 16]])))
    (let [[v [_ da db]] (test-op join-2-along [[9 10 13] [11 12 15]] 1 [[1 2] [3 4]] [[5] [7]])]
      (is (nd= v [[1 2 5] [3 4 7]]))
      (is (nd= da [[9 10] [11 12]]))
      (is (nd= db [[13] [15]]))))
  (testing "joins tensors"
    (let [a [[[1 2 3] [4 5 6]] [[7 8 9] [10 11 12]]]
          b [[[13 14 15] [16 17 18]] [[19 20 21] [22 23 24]]]
          expected-da [[[25 26 27] [28 29 30]] [[31 32 33] [34 35 36]]]
          expected-db [[[37 38 39] [40 41 42]] [[43 44 45] [46 47 48]]]]
      (let [[v [_ da db]] (test-op join-2-along [[[25 26 27] [28 29 30]]
                                                 [[31 32 33] [34 35 36]]
                                                 [[37 38 39] [40 41 42]]
                                                 [[43 44 45] [46 47 48]]] 0 a b)]
        (is (nd= v [[[1 2 3] [4 5 6]] [[7 8 9] [10 11 12]] [[13 14 15] [16 17 18]] [[19 20 21] [22 23 24]]]))
        (is (nd= da expected-da))
        (is (nd= db expected-db)))
      (let [[v [_ da db]] (test-op join-2-along [[[25 26 27] [28 29 30] [37 38 39] [40 41 42]]
                                                 [[31 32 33] [34 35 36] [43 44 45] [46 47 48]]] 1 a b)]
        (is (nd= v [[[1 2 3] [4 5 6] [13 14 15] [16 17 18]] [[7 8 9] [10 11 12] [19 20 21] [22 23 24]]]))
        (is (nd= da expected-da))
        (is (nd= db expected-db)))
      (let [[v [_ da db]] (test-op join-2-along [[[25 26 27 37 38 39] [28 29 30 40 41 42]]
                                                 [[31 32 33 43 44 45] [34 35 36 46 47 48]]] 2 a b)]
        (is (nd= v [[[1 2 3 13 14 15] [4 5 6 16 17 18]] [[7 8 9 19 20 21] [10 11 12 22 23 24]]]))
        (is (nd= da expected-da))
        (is (nd= db expected-db))))))

(deftest join-along-test
  (testing "non-scalar dimension throws an error"
    (is (thrown? Exception (evaluate (join-along [1 2] [3 4] [5 6])))))
  (testing "empty tensor list throws an error"
    (is (thrown? Exception (evaluate (join-along 0)))))
  (testing "1 operand"
    (let [[v [_ da]] (test-op join-along [[4 5 6]] 0 [[1 2 3]])]
      (is (nd= v [[1 2 3]]))
      (is (nd= da [[4 5 6]]))))
  (testing "2 operand"
    (let [[v [_ da db]] (test-op join-along [[7 8 9] [10 11 12]] 0 [[1 2 3]] [[4 5 6]])]
      (is (nd= v [[1 2 3] [4 5 6]]))
      (is (nd= da [[7 8 9]]))
      (is (nd= db [[10 11 12]]))))
  (testing "3 operands"
    (let [[v [_ da db dc]] (test-op join-along [[10 11 12] [13 14 15] [16 17 18]] 0 [[1 2 3]] [[4 5 6]] [[7 8 9]])]
      (is (nd= v [[1 2 3] [4 5 6] [7 8 9]]))
      (is (nd= da [[10 11 12]]))
      (is (nd= db [[13 14 15]]))
      (is (nd= dc [[16 17 18]])))
    (let [[v [_ da db dc]] (test-op join-along [[10 11 12 13 14 15 16 17 18]] 1 [[1 2 3]] [[4 5 6]] [[7 8 9]])]
      (is (nd= v [[1 2 3 4 5 6 7 8 9]]))
      (is (nd= da [[10 11 12]]))
      (is (nd= db [[13 14 15]]))
      (is (nd= dc [[16 17 18]])))))

(deftest shift-all-by-test
  #_(testing "shifts scalars"
      (let [[v [da _]] (test-op shift-all-by 7 3 [])]
        (is (nd= v 3))
        (is (nd= da 7))))
  (testing "shifts vectors"
    (let [[v [da _]] (test-op shift-all-by [4 5 6] [1 2 3] [0])]
      (is (nd= v [1 2 3]))
      (is (nd= da [4 5 6])))
    (let [[v [da _]] (test-op shift-all-by [4 5 6] [1 2 3] [1])]
      (is (nd= v [2 3 0]))
      (is (nd= da [0 4 5])))
    (let [[v [da _]] (test-op shift-all-by [4 5 6] [1 2 3] [-2])]
      (is (nd= v [0 0 1]))
      (is (nd= da [6 0 0])))
    (let [[v [da _]] (test-op shift-all-by [4 5 6] [1 2 3] [50])]
      (is (nd= v [0 0 0]))
      (is (nd= da [0 0 0]))))
  (testing "shifts matrices"
    (let [[v [da _]] (test-op shift-all-by [[7 8 9] [10 11 12]] [[1 2 3] [4 5 6]] [-1 2])]
      (is (nd= v [[0 0 0] [3 0 0]]))
      (is (nd= da [[0 0 10] [0 0 0]])))
    (let [[v [da _]] (test-op shift-all-by [[7 8 9] [10 11 12]] [[1 2 3] [4 5 6]] [0 -1])]
      (is (nd= v [[0 1 2] [0 4 5]]))
      (is (nd= da [[8 9 0] [11 12 0]]))))
  (testing "shifts tensors"
    (let [[v [da _]] (test-op shift-all-by [[[13 14 15] [16 17 18]] [[19 20 21] [22 23 24]]]
                              [[[1 2 3] [4 5 6]] [[7 8 9] [10 11 12]]]
                              [-1 1 -1])]
      (is (nd= v [[[0 0 0] [0 0 0]] [[0 4 5] [0 0 0]]]))
      (is (nd= da [[[0 0 0] [20 21 0]] [[0 0 0] [0 0 0]]])))))

(deftest shift-dim-by-test
  (testing "shifts vectors"
    (let [[v [da]] (test-op shift-dim-by [4 5 6] [1 2 3] 0 1)]
      (is (nd= v [2 3 0]))
      (is (nd= da [0 4 5]))))
  (testing "shifts matrices"
    (let [[v [da _]] (test-op shift-dim-by [[7 8 9] [10 11 12]] [[1 2 3] [4 5 6]] 1 -1)]
      (is (nd= v [[0 1 2] [0 4 5]]))
      (is (nd= da [[8 9 0] [11 12 0]])))))

(deftest shift-test
  (testing "works with `shifts` argument"
    (let [[v [da _]] (test-op shift [[7 8 9] [10 11 12]] [[1 2 3] [4 5 6]] [-1 2])]
      (is (nd= v [[0 0 0] [3 0 0]]))
      (is (nd= da [[0 0 10] [0 0 0]]))))
  (testing "works with single-dimension arguments"
    (let [[v [da _]] (test-op shift-dim-by [[7 8 9] [10 11 12]] [[1 2 3] [4 5 6]] 1 -1)]
      (is (nd= v [[0 1 2] [0 4 5]]))
      (is (nd= da [[8 9 0] [11 12 0]])))))

(deftest select-range-test
  (testing "scalars"
    (let [[v da] (test-op-with-args select-range 7 3 [])]
      (is (nd= v 3))
      (is (nd= da 7))))
  (testing "vectors"
    (let [[v da] (test-op-with-args select-range 4 [1 2 3] [0])]
      (is (nd= v 1))
      (is (nd= da [4 0 0])))
    (let [[v da] (test-op-with-args select-range [4 5] [1 2 3] [[1 3]])]
      (is (nd= v [2 3]))
      (is (nd= da [0 4 5])))
    (let [[v da] (test-op-with-args select-range [4 5 6] [1 2 3] [:all])]
      (is (nd= v [1 2 3]))
      (is (nd= da [4 5 6])))
    (let [[v da] (test-op-with-args select-range [4 5 6] [1 2 3] [:all])]
      (is (nd= v [1 2 3]))
      (is (nd= da [4 5 6])))
    (let [[v da] (test-op-with-args select-range 4 [1 2 3] [:first])]
      (is (nd= v 1))
      (is (nd= da [4 0 0])))
    (let [[v da] (test-op-with-args select-range 4 [1 2 3] [:last])]
      (is (nd= v 3))
      (is (nd= da [0 0 4])))
    (let [[v da] (test-op-with-args select-range [4 5] [1 2 3] [:butlast])]
      (is (nd= v [1 2]))
      (is (nd= da [4 5 0])))
    (let [[v da] (test-op-with-args select-range [4 5] [1 2 3] [:rest])]
      (is (nd= v [2 3]))
      (is (nd= da [0 4 5]))))
  (testing "selecting a full slice while eliminating a dimension"
    (let [[v da] (test-op-with-args select-range [7 8 9] [[1 2 3] [4 5 6]] [0 :all])]
      (is (nd= v [1 2 3]))
      (is (nd= da [[7 8 9] [0 0 0]])))
    (let [[v da] (test-op-with-args select-range 13 [[[1 2 3] [4 5 6]] [[7 8 9] [10 11 12]]] [1 0 2])]
      (is (nd= v 9))
      (is (nd= da [[[0 0 0] [0 0 0]] [[0 0 13] [0 0 0]]]))))
  (testing "selecting a set of slices"
    (let [[v da] (test-op-with-args select-range [[7 8 9]] [[1 2 3] [4 5 6]] [[0 1] :all])]
      (is (nd= v [[1 2 3]]))
      (is (nd= da [[7 8 9] [0 0 0]])))
    (let [[v da] (test-op-with-args select-range [[7 8]] [[1 2 3] [4 5 6]] [[1 2] [1 3]])]
      (is (nd= v [[5 6]]))
      (is (nd= da [[0 0 0] [0 7 8]])))
    (let [[v da] (test-op-with-args select-range [[13 14] [15 16]] [[[1 2 3] [4 5 6]] [[7 8 9] [10 11 12]]] [1 [0 2] [0 2]])]
      (is (nd= v [[7 8] [10 11]]))
      (is (nd= da [[[0 0 0] [0 0 0]] [[13 14 0] [15 16 0]]]))))
  (testing "selecting slices with a keyword"
    (let [[v da] (test-op-with-args select-range [7 8 9] [[1 2 3] [4 5 6]] [:first :all])]
      (is (nd= v [1 2 3]))
      (is (nd= da [[7 8 9] [0 0 0]])))
    (let [[v da] (test-op-with-args select-range [7] [[1 2 3] [4 5 6]] [:butlast :last])]
      (is (nd= v [3]))
      (is (nd= da [[0 0 7] [0 0 0]])))
    (let [[v da] (test-op-with-args select-range [7] [[1 2 3] [4 5 6]] [:rest :first])]
      (is (nd= v [4]))
      (is (nd= da [[0 0 0] [7 0 0]])))
    (let [[v da] (test-op-with-args select-range [7 8] [[1 2 3] [4 5 6]] [:all :first])]
      (is (nd= v [1 4]))
      (is (nd= da [[7 0 0] [8 0 0]])))))

(deftest make-shape-test
  (testing "backpropagates nil"
    (let [[v [da]] (test-op make-shape [2] 1)]
      (is (= v [1]))
      (is (nil? da))))
  (testing "returns correct shapes"
    (is (evaluates-to? (make-shape) []))
    (is (evaluates-to? (make-shape 1 2) [1 2]))
    (is (evaluates-to? (make-shape 1 2 3) [1 2 3]))
    (is (evaluates-to? (make-shape 1 2 3 4 5 6 7 8 9 10) [1 2 3 4 5 6 7 8 9 10]))))

(deftest pad-with-test
  (testing "vectors"
    (let [[v [dnd dfill]] (test-op pad-with [3 4] [1 2] 7 0)]
      (is (nd= v [1 2]))
      (is (nd= dnd [3 4]))
      (is (nd= dfill 0)))
    (let [[v [dnd dfill]] (test-op pad-with [3 4 5 6] [1 2] 7 1)]
      (is (nd= v [7 1 2 7]))
      (is (nd= dnd [4 5]))
      (is (nd= dfill 9)))
    (let [[v [dnd dfill]] (test-op pad-with [3 4 5 6 7 8] [1 2] 7 [2])]
      (is (nd= v [7 7 1 2 7 7]))
      (is (nd= dnd [5 6]))
      (is (nd= dfill 22)))
    (let [[v [dnd dfill]] (test-op pad-with [3 4 5 6 7] [1 2] 7 [[1 2]])]
      (is (nd= v [7 1 2 7 7]))
      (is (nd= dnd [4 5]))
      (is (nd= dfill 16))))
  (testing "matrices"
    (let [[v [dnd dfill]] (test-op pad-with [[5 6] [7 8]] [[1 2] [3 4]] 2 0)]
      (is (nd= v [[1 2] [3 4]]))
      (is (nd= dnd [[5 6] [7 8]]))
      (is (nd= dfill 0)))
    (let [[v [dnd dfill]] (test-op pad-with [[5 6 7 8] [9 10 11 12] [13 14 15 16] [17 18 19 20]] [[1 2] [3 4]] 2 1)]
      (is (nd= v [[2 2 2 2] [2 1 2 2] [2 3 4 2] [2 2 2 2]]))
      (is (nd= dnd [[10 11] [14 15]]))
      (is (nd= dfill 150)))
    (let [[v [dnd dfill]] (test-op pad-with [[5 6 7 8] [9 10 11 12]] [[1 2] [3 4]] 9 [0 1])]
      (is (nd= v [[9 1 2 9] [9 3 4 9]]))
      (is (nd= dnd [[6 7] [10 11]]))
      (is (nd= dfill 34)))
    (let [[v [dnd dfill]] (test-op pad-with [[5 6] [7 8] [9 10] [11 12]] [[1 2] [3 4]] 9 [1 0])]
      (is (nd= v [[9 9] [1 2] [3 4] [9 9]]))
      (is (nd= dnd [[7 8] [9 10]]))
      (is (nd= dfill 34)))
    (let [[v [dnd dfill]] (test-op pad-with [[5 6 7 8 9] [10 11 12 13 14] [15 16 17 18 19]] [[1 2] [3 4]] 1 [[1 0] [1 2]])]
      (is (nd= v [[1 1 1 1 1] [1 1 2 1 1] [1 3 4 1 1]]))
      (is (nd= dnd [[11 12] [16 17]]))
      (is (nd= dfill 124))))
  (testing "tensors"
    (let [[v [dnd dfill]] (test-op pad-with [[[5] [6]] [[7] [8]]] [[[1] [2]] [[3] [4]]] 7 0)]
      (is (nd= v [[[1] [2]] [[3] [4]]]))
      (is (nd= dnd [[[5] [6]] [[7] [8]]]))
      (is (nd= dfill 0)))
    (let [[v [dnd dfill]] (test-op pad-with [[[0 1 2] [1 2 0] [3 4 0] [1 1 1]] [[2 8 1] [0 1 0] [2 0 2] [3 3 1]]
                                             [[1 0 1] [2 2 3] [0 4 0] [3 4 5]] [[0 2 1] [1 2 3] [2 1 4] [0 0 0]]]
                                   [[[1] [2]] [[3] [4]]] 7 1)]
      (is (nd= v [[[7 7 7] [7 7 7] [7 7 7] [7 7 7]] [[7 7 7] [7 1 7] [7 2 7] [7 7 7]]
                  [[7 7 7] [7 3 7] [7 4 7] [7 7 7]] [[7 7 7] [7 7 7] [7 7 7] [7 7 7]]]))
      (is (nd= dnd [[[1] [0]] [[2] [4]]]))
      (is (nd= dfill 73)))
    (let [[v [dnd dfill]] (test-op pad-with [[[0 1 2 1 2] [0 3 4 0 1] [1 2 8 1 0] [1 0 2 0 2]]
                                             [[3 3 1 1 0] [1 2 2 3 0] [0 4 0 3 4] [5 0 2 1 1]]]
                                   [[[1] [2]] [[3] [4]]] 7 [0 1 2])]
      (is (nd= v [[[7 7 7 7 7] [7 7 1 7 7] [7 7 2 7 7] [7 7 7 7 7]]
                  [[7 7 7 7 7] [7 7 3 7 7] [7 7 4 7 7] [7 7 7 7 7]]]))
      (is (nd= dnd [[[4] [8]] [[2] [0]]]))
      (is (nd= dfill 53))))
  (testing "throws exception on dimension mismatch"
    (is (thrown? Exception (evaluate (pad-with [1 2] 0 [1 2]))))
    (is (thrown? Exception (evaluate (pad-with [[1 2] [3 4]] 0 [1]))))
    (is (thrown? Exception (evaluate (pad-with [[1 2] [3 4]] 0 [[1 1] [2 2] [3 3]]))))))

(deftest pad-test
  (testing "vector"
    (let [[v [dnd]] (test-op pad [3 4 5 6 7] [1 2] [[1 2]])]
      (is (nd= v [0 1 2 0 0]))
      (is (nd= dnd [4 5]))))
  (testing "matrix"
    (let [[v [dnd]] (test-op pad [[1 2 3 4] [5 6 7 8] [9 10 11 12] [13 14 15 16]] [[1 2] [3 4]] 1)]
      (is (nd= v [[0 0 0 0] [0 1 2 0] [0 3 4 0] [0 0 0 0]]))
      (is (nd= dnd [[6 7] [10 11]]))))
  (testing "tensor"
    (let [[v [dnd]] (test-op pad [[[1 2 3]] [[4 5 6]] [[7 8 9]]] [[[1]]] [1 0 1])]
      (is (nd= v [[[0 0 0]] [[0 1 0]] [[0 0 0]]]))
      (is (nd= dnd [[[5]]])))))

(deftest zeros-test
  (testing "backpropagates nil"
    (let [[_ [grad]] (test-op zeros [[1 2 3] [4 5 6]] [2 3])]
      (is (nil? grad))))
  (testing "returns 0-filled tensors of the correct size"
    #_(evaluates-to? (zeros []) 0)
    (evaluates-to? (zeros [3]) [0 0 0])
    (evaluates-to? (zeros [2 3]) [[0 0 0] [0 0 0]])
    (evaluates-to? (zeros [2 2 3]) [[[0 0 0] [0 0 0]] [[0 0 0] [0 0 0]]])))

(deftest ones-test
  (testing "backpropagates nil"
    (let [[_ [grad]] (test-op ones [[1 2 3] [4 5 6]] [2 3])]
      (is (nil? grad))))
  (testing "returns 1-filled tensors of the correct size"
    #_(evaluates-to? (ones []) 1)
    (evaluates-to? (ones [3]) [1 1 1])
    (evaluates-to? (ones [2 3]) [[1 1 1] [1 1 1]])
    (evaluates-to? (ones [2 2 3]) [[[1 1 1] [1 1 1]] [[1 1 1] [1 1 1]]])))

(deftest random-uniform-test
  (set-rng-seed! 0)
  (testing "backpropagates nil"
    (let [[_ [grad]] (test-op random-uniform [[1 2 3] [4 5 6]] [2 3])]
      (is (nil? grad))))
  (check-tensor-creator #(evaluate (random-uniform %)))
  (testing "produces tensors with the expected statistical properties"
    (let [n (evaluate (random-uniform [10000]))]
      (is (>= (m/scalar->number (m/emin n)) 0))
      (is (< (m/scalar->number (m/emax n)) 1))
      (is (about= (m/emean n) 0.5 0.01))
      (is (about= (m/estdev n) (uniform-sd 0 1) 0.01)))))

(deftest random-normal-test
  (set-rng-seed! 0)
  (testing "backpropagates nil"
    (let [[_ [grad]] (test-op random-normal [[1 2 3] [4 5 6]] [2 3])]
      (is (nil? grad))))
  (check-tensor-creator #(evaluate (random-normal %)))
  (testing "produces tensors with the expected statistical properties"
    (let [n (evaluate (random-normal [10000]))]
      (is (about= (m/emean n) 0 0.1))
      (is (about= (m/estdev n) 1 0.1)))))

(deftest and-2-test
  (check-op-backpropagates-nil and-2)
  (testing "scalar && scalar"
    (evaluates-to? (and-2 0 0) 0)
    (evaluates-to? (and-2 0 1) 0)
    (evaluates-to? (and-2 1 0) 0)
    (evaluates-to? (and-2 1 1) 1)
    (evaluates-to? (and-2 3 7) 1)
    (evaluates-to? (and-2 3 0) 0))
  (testing "scalar && vector"
    (evaluates-to? (and-2 [0 1] 0) [0 0])
    (evaluates-to? (and-2 [0 1] 1) [0 1])
    (evaluates-to? (and-2 1 [0 1]) [0 1]))
  (testing "scalar && matrix"
    (evaluates-to? (and-2 [[0 1] [1 0]] 0) [[0 0] [0 0]])
    (evaluates-to? (and-2 1 [[0 1] [1 0]]) [[0 1] [1 0]]))
  (testing "matrix && vector"
    (evaluates-to? (and-2 [[0 1] [1 0]] [0 1]) [[0 1] [0 0]])
    (evaluates-to? (and-2 [1 1] [[0 1] [1 0]]) [[0 1] [1 0]]))
  (testing "matrix && matrix"
    (evaluates-to? (and-2 [[0 1] [1 0]] [[0 1] [1 1]]) [[0 1] [1 0]])
    (evaluates-to? (and-2 [[0 3] [1 0]] [[0 1] [1 7]]) [[0 1] [1 0]])))

(deftest tensor-and-test
  (testing "0 operands"
    (evaluates-to? (tensor-and) 1))
  (testing "1 operands"
    #_(evaluates-to? (tensor-and []) [])
    (evaluates-to? (tensor-and 1) 1)
    (evaluates-to? (tensor-and 3) 1)
    (evaluates-to? (tensor-and 0) 0)
    (evaluates-to? (tensor-and [1 0 3]) [1 0 1])
    (evaluates-to? (tensor-and [[0 1 2] [1 0 3]]) [[0 1 1] [1 0 1]]))
  (testing "2 operands"
    (evaluates-to? (tensor-and [1 0 1 0 3] [1 1 0 0 1]) [1 0 0 0 1])
    (evaluates-to? (tensor-and [[1 0 3] [3 1 0]] [1 0 1]) [[1 0 1] [1 0 0]])
    (evaluates-to? (tensor-and [1 0 1] [[1 0 3] [3 1 0]]) [[1 0 1] [1 0 0]])
    (evaluates-to? (tensor-and [[1 0] [0 3]] [[1 1] [0 0]]) [[1 0] [0 0]])
    (evaluates-to? (tensor-and [[[0 1 0] [3 0 7]] [[1 1 1] [0 0 0]]] [[1 1 0] [0 0 1]])
                   [[[0 1 0] [0 0 1]] [[1 1 0] [0 0 0]]]))
  (testing "3 operands"
    (evaluates-to? (tensor-and 1 1 1) 1)
    (evaluates-to? (tensor-and 1 0 1) 0)
    (evaluates-to? (tensor-and 0 1 1) 0)
    (evaluates-to? (tensor-and 0 0 0) 0)
    (evaluates-to? (tensor-and [[1 1] [0 0]] [[0 1] [1 0]] [[0 1] [0 1]]) [[0 1] [0 0]])))

(deftest eq-2-test
  (check-op-backpropagates-nil eq-2)
  (testing "scalar = scalar"
    (evaluates-to? (eq-2 3 3) 1)
    (evaluates-to? (eq-2 3 7) 0))
  (testing "scalar = vector"
    (evaluates-to? (eq-2 [3 2] [[1 2] [3 4]]) [[0 1] [1 0]]))
  (testing "scalar = matrix"
    (evaluates-to? (eq-2 3 [[1 2] [3 4]]) [[0 0] [1 0]]))
  (testing "matrix = matrix"
    (evaluates-to? (eq-2 [[1 2] [3 4]] [[1 3] [2 4]]) [[1 0] [0 1]])))

(deftest ne-2-test
  (check-op-backpropagates-nil ne-2)
  (testing "scalar != scalar"
    (evaluates-to? (ne-2 3 3) 0)
    (evaluates-to? (ne-2 3 7) 1))
  (testing "scalar != vector"
    (evaluates-to? (ne-2 [3 2] [[1 2] [3 4]]) [[1 0] [0 1]]))
  (testing "scalar != matrix"
    (evaluates-to? (ne-2 3 [[1 2] [3 4]]) [[1 1] [0 1]]))
  (testing "matrix != matrix"
    (evaluates-to? (ne-2 [[1 2] [3 4]] [[1 3] [2 4]]) [[0 1] [1 0]])))

(deftest eq-test
  (testing "0 operands"
    (is (thrown? Exception (evaluate (eq)))))
  #_(testing "1 operand"
      (evaluates-to? (eq 0) 1)
      (evaluates-to? (eq [1 2 3]) [1 1 1])
      (evaluates-to? (eq [[1 2] [3 4]]) [[1 1] [1 1]]))
  (testing "2 operands"
    (evaluates-to? (eq 3 7) 0)
    (evaluates-to? (eq 7 7) 1)
    (evaluates-to? (eq [7 3] 3) [0 1])
    (evaluates-to? (eq 3 [7 3]) [0 1])
    (evaluates-to? (eq [[1 2] [3 4]] 3) [[0 0] [1 0]])
    (evaluates-to? (eq 3 [[1 2] [3 4]]) [[0 0] [1 0]])
    (evaluates-to? (eq [[1 2] [3 4]] [3 2]) [[0 1] [1 0]])
    (evaluates-to? (eq [3 2] [[1 2] [3 4]]) [[0 1] [1 0]])
    (evaluates-to? (eq [[1 2] [3 4]] [[1 3] [3 2]]) [[1 0] [1 0]])
    (evaluates-to? (eq [[[1 2] [3 4]] [[5 6] [7 8]]] [[1 6] [3 8]]) [[[1 0] [1 0]] [[0 1] [0 1]]]))
  (testing "3 operands"
    (evaluates-to? (eq 3 7 3) 0)
    (evaluates-to? (eq 7 3 3) 0)
    (evaluates-to? (eq 3 3 7) 0)
    (evaluates-to? (eq 3 3 3) 1)
    (evaluates-to? (eq [1 2 3] [1 4 3] [5 4 3]) [0 0 1])
    (evaluates-to? (eq [1 2 3] 2 [1 2 4]) [0 1 0])
    (evaluates-to? (eq [[1.0 2.2 3.3] [4.4 5.5 6.6]]
                       [[1.0 2.3 3.3] [4.0 5.5 6.0]]
                       [[1.0 2.2 3.4] [4.0 5.5 6.0]])
                   [[1 0 0] [0 1 0]])))

(deftest ne-test
  (testing "0 operands"
    (is (thrown? Exception (evaluate (ne)))))
  #_(testing "1 operand"
      (evaluates-to? (ne 0) 0)
      (evaluates-to? (ne [1 2 3]) [0 0 0])
      (evaluates-to? (ne [[1 2] [3 4]]) [[0 0] [0 0]]))
  (testing "2 operands"
    (evaluates-to? (ne 3 7) 1)
    (evaluates-to? (ne 7 7) 0)
    (evaluates-to? (ne [7 3] 3) [1 0])
    (evaluates-to? (ne 3 [7 3]) [1 0])
    (evaluates-to? (ne [[1 2] [3 4]] 3) [[1 1] [0 1]])
    (evaluates-to? (ne 3 [[1 2] [3 4]]) [[1 1] [0 1]])
    (evaluates-to? (ne [[1 2] [3 4]] [3 2]) [[1 0] [0 1]])
    (evaluates-to? (ne [3 2] [[1 2] [3 4]]) [[1 0] [0 1]])
    (evaluates-to? (ne [[1 2] [3 4]] [[1 3] [3 2]]) [[0 1] [0 1]])
    (evaluates-to? (ne [[[1 2] [3 4]] [[5 6] [7 8]]] [[1 6] [3 8]]) [[[0 1] [0 1]] [[1 0] [1 0]]]))
  (testing "3 operands"
    (evaluates-to? (ne 3 7 3) 1)
    (evaluates-to? (ne 7 3 3) 1)
    (evaluates-to? (ne 3 3 7) 1)
    (evaluates-to? (ne 3 3 3) 0)
    (evaluates-to? (ne [1 2 3] [1 4 3] [5 4 3]) [1 1 0])
    (evaluates-to? (ne [1 2 3] 2 [1 2 4]) [1 0 1])
    (evaluates-to? (ne [[1.0 2.2 3.3] [4.4 5.5 6.6]]
                       [[1.0 2.3 3.3] [4.0 5.5 6.0]]
                       [[1.0 2.2 3.4] [4.0 5.5 6.0]])
                   [[0 1 1] [1 0 1]])))

(deftest or-2-test
  (check-op-backpropagates-nil or-2)
  (testing "scalar or scalar"
    (evaluates-to? (or-2 0 1) 1)
    (evaluates-to? (or-2 1 0) 1)
    (evaluates-to? (or-2 1 1) 1)
    (evaluates-to? (or-2 0 0) 0))
  (testing "scalar or matrix"
    (is (about= (evaluate (or-2 [[0 1] [1 0]] 1)) [[1 1] [1 1]]))
    (is (about= (evaluate (or-2 [[0 1] [1 0]] 0)) [[0 1] [1 0]])))
  (testing "vector or vector"
    (is (about= (evaluate (or-2 [[0 1] [1 0]] [0 1])) [[0 1] [1 1]])))
  (testing "matrix or matrix"
    (is (about= (evaluate (or-2 [[0 1] [1 0]] [[1 1] [0 0]])) [[1 1] [1 0]]))))

(deftest tensor-or-test
  #_(testing "0 operands"
      (evaluates-to? (tensor-or) 0))
  (testing "1 operands"
    #_(evaluates-to? (tensor-or []) [])
    (evaluates-to? (tensor-or 1) 1)
    (evaluates-to? (tensor-or 3) 1)
    (evaluates-to? (tensor-or 0) 0)
    (evaluates-to? (tensor-or [1 0 3]) [1 0 1])
    (evaluates-to? (tensor-or [[0 1 2] [1 0 3]]) [[0 1 1] [1 0 1]]))
  (testing "2 operands"
    (evaluates-to? (tensor-or [1 0 1 0 3] [1 1 0 0 1]) [1 1 1 0 1])
    (evaluates-to? (tensor-or [[1 0 3] [3 1 0]] [1 0 0]) [[1 0 1] [1 1 0]])
    (evaluates-to? (tensor-or [1 0 0] [[1 0 3] [3 1 0]]) [[1 0 1] [1 1 0]])
    (evaluates-to? (tensor-or [[1 0] [0 3]] [[1 1] [0 0]]) [[1 1] [0 1]])
    (evaluates-to? (tensor-or [[[0 1 0] [3 0 7]] [[1 1 1] [0 0 0]]] [[1 1 0] [0 0 1]])
                   [[[1 1 0] [1 0 1]] [[1 1 1] [0 0 1]]]))
  (testing "3 operands"
    (evaluates-to? (tensor-or 1 1 1) 1)
    (evaluates-to? (tensor-or 1 0 1) 1)
    (evaluates-to? (tensor-or 0 1 1) 1)
    (evaluates-to? (tensor-or 0 0 1) 1)
    (evaluates-to? (tensor-or 0 0 0) 0)
    (evaluates-to? (tensor-or [[1 1] [0 0]] [[0 1] [1 0]] [[0 1] [0 0]]) [[1 1] [1 0]])))

(deftest tensor-not-test
  (testing "backpropagates nil"
    (let [[_ [grad]] (test-op tensor-not [[1 2 3] [4 5 6]] [[0 1 0] [1 0 1]])]
      (is (nil? grad))))
  (testing "scalars"
    (evaluates-to? (tensor-not 1) 0)
    (evaluates-to? (tensor-not 3) 0)
    (evaluates-to? (tensor-not 0) 1))
  (testing "vectors"
    (evaluates-to? (tensor-not [-3 0.4 1 0]) [0 0 0 1]))
  (testing "matrices"
    (evaluates-to? (tensor-not [[0 1] [3 0.0]]) [[1 0] [0 1]]))
  (testing "tensors"
    (evaluates-to? (tensor-not [[[1 0] [3 4]] [[0 1] [-2.1 -0.0]]]) [[[0 1] [0 0]] [[1 0] [0 1]]])))

(deftest gt-2-test
  (check-op-backpropagates-nil gt-2)
  (testing "scalar > scalar"
    (evaluates-to? (gt-2 1 0) 1)
    (evaluates-to? (gt-2 0 1) 0)
    (evaluates-to? (gt-2 0 0) 0)
    (evaluates-to? (gt-2 3 -2) 1)
    (evaluates-to? (gt-2 -2 3) 0))
  (testing "scalar > vector"
    (evaluates-to? (gt-2 2 [-3 2 3]) [1 0 0]))
  (testing "scalar > matrix"
    (evaluates-to? (gt-2 2 [[-3 2] [0 3]]) [[1 0] [1 0]]))
  (testing "vector > scalar"
    (evaluates-to? (gt-2 [-3 2 3] 2) [0 0 1]))
  (testing "vector > matrix"
    (evaluates-to? (gt-2 [-2 2] [[-3 2] [0 3]]) [[1 0] [0 0]]))
  (testing "matrix > scalar"
    (evaluates-to? (gt-2 [[-3 2] [0 3]] 2) [[0 0] [0 1]]))
  (testing "matrix > vector"
    (evaluates-to? (gt-2 [[-3 2] [0 3]] [-2 2]) [[0 0] [1 1]]))
  (testing "matrix > matrix"
    (evaluates-to? (gt-2 [[-3 2] [0 3]] [[-5 3.14] [0 2]]) [[1 0] [0 1]])))

(deftest gt-test
  #_(testing "0 operands"
      (is (thrown? Exception (evaluate (gt)))))
  #_(testing "1 operand"
      (evaluates-to? (gt 0) 1)
      (evaluates-to? (gt [1 2 3]) [1 1 1])
      (evaluates-to? (gt [[1 2] [3 4]]) [[1 1] [1 1]]))
  (testing "2 operands"
    (evaluates-to? (gt 3 7) 0)
    (evaluates-to? (gt 7 3) 1)
    (evaluates-to? (gt 7 7) 0)
    (evaluates-to? (gt [7 3 0] 3) [1 0 0])
    (evaluates-to? (gt 3 [7 3 0]) [0 0 1])
    (evaluates-to? (gt [[1 2] [3 4]] 3) [[0 0] [0 1]])
    (evaluates-to? (gt 3 [[1 2] [3 4]]) [[1 1] [0 0]])
    (evaluates-to? (gt [[1 2] [3 4]] [3 2]) [[0 0] [0 1]])
    (evaluates-to? (gt [3 2] [[1 2] [3 4]]) [[1 0] [0 0]])
    (evaluates-to? (gt [[1 2] [3 4]] [[1 3] [3 2]]) [[0 0] [0 1]])
    (evaluates-to? (gt [[[1 2] [3 4]] [[5 6] [7 8]]] [[1 6] [3 8]]) [[[0 0] [0 0]] [[1 0] [1 0]]]))
  (testing "3 operands"
    (evaluates-to? (gt 3 4 5) 0)
    (evaluates-to? (gt 5 4 3) 1)
    (evaluates-to? (gt 5 3 4) 0)
    (evaluates-to? (gt 3 5 4) 0)
    (evaluates-to? (gt 5 4 4) 0)
    (evaluates-to? (gt 5 5 4) 0)
    (evaluates-to? (gt [3 5 5 4 5 5] [4 4 3 5 4 5] [5 3 4 4 4 4]) [0 1 0 0 0 0])
    (evaluates-to? (gt [3 5 6 3 5 4] 4 [5 3 5 3 4 3]) [0 1 0 0 0 0])
    (evaluates-to? (gt [[1.0 2.4 3.3] [4.4 5.5 6.6]]
                       [[1.0 2.3 3.3] [4.5 5.5 6.0]]
                       [[1.0 2.2 3.4] [4.6 5.5 6.0]])
                   [[0 1 0] [0 0 0]])))

(deftest ge-2-test
  (check-op-backpropagates-nil ge-2)
  (testing "scalar >= scalar"
    (evaluates-to? (ge-2 1 0) 1)
    (evaluates-to? (ge-2 0 1) 0)
    (evaluates-to? (ge-2 0 0) 1)
    (evaluates-to? (ge-2 3 -2) 1)
    (evaluates-to? (ge-2 -2 3) 0))
  (testing "scalar >= vector"
    (evaluates-to? (ge-2 2 [-3 2 3]) [1 1 0]))
  (testing "scalar >= matrix"
    (evaluates-to? (ge-2 2 [[-3 2] [0 3]]) [[1 1] [1 0]]))
  (testing "vector >= scalar"
    (evaluates-to? (ge-2 [-3 2 3] 2) [0 1 1]))
  (testing "vector >= matrix"
    (evaluates-to? (ge-2 [-2 2] [[-3 2] [0 3]]) [[1 1] [0 0]]))
  (testing "matrix >= scalar"
    (evaluates-to? (ge-2 [[-3 2] [0 3]] 2) [[0 1] [0 1]]))
  (testing "matrix >= vector"
    (evaluates-to? (ge-2 [[-3 2] [0 3]] [-2 2]) [[0 1] [1 1]]))
  (testing "matrix >= matrix"
    (evaluates-to? (ge-2 [[-3 2] [0 3]] [[-5 3.14] [0 2]]) [[1 0] [1 1]])))

(deftest ge-test
  #_(testing "0 operands"
      (is (thrown? Exception (evaluate (ge)))))
  #_(testing "1 operand"
      (evaluates-to? (ge 0) 1)
      (evaluates-to? (ge [1 2 3]) [1 1 1])
      (evaluates-to? (ge [[1 2] [3 4]]) [[1 1] [1 1]]))
  (testing "2 operands"
    (evaluates-to? (ge 3 7) 0)
    (evaluates-to? (ge 7 3) 1)
    (evaluates-to? (ge 7 7) 1)
    (evaluates-to? (ge [7 3 0] 3) [1 1 0])
    (evaluates-to? (ge 3 [7 3 0]) [0 1 1])
    (evaluates-to? (ge [[1 2] [3 4]] 3) [[0 0] [1 1]])
    (evaluates-to? (ge 3 [[1 2] [3 4]]) [[1 1] [1 0]])
    (evaluates-to? (ge [[1 2] [3 4]] [3 2]) [[0 1] [1 1]])
    (evaluates-to? (ge [3 2] [[1 2] [3 4]]) [[1 1] [1 0]])
    (evaluates-to? (ge [[1 2] [3 4]] [[1 3] [3 2]]) [[1 0] [1 1]])
    (evaluates-to? (ge [[[1 2] [3 4]] [[5 6] [7 8]]] [[1 6] [3 8]]) [[[1 0] [1 0]] [[1 1] [1 1]]]))
  (testing "3 operands"
    (evaluates-to? (ge 3 4 5) 0)
    (evaluates-to? (ge 5 4 3) 1)
    (evaluates-to? (ge 5 3 4) 0)
    (evaluates-to? (ge 3 5 4) 0)
    (evaluates-to? (ge 5 4 4) 1)
    (evaluates-to? (ge 5 5 4) 1)
    (evaluates-to? (ge [3 5 5 4 5 5] [4 4 3 5 4 5] [5 3 4 4 4 4]) [0 1 0 0 1 1])
    (evaluates-to? (ge [3 5 6 3 5 4] 4 [5 3 5 3 4 3]) [0 1 0 0 1 1])
    (evaluates-to? (ge [[1.0 2.4 3.3] [4.4 5.5 6.6]]
                       [[1.0 2.3 3.3] [4.5 5.5 6.0]]
                       [[1.0 2.2 3.4] [4.6 5.5 6.0]])
                   [[1 1 0] [0 1 1]])))

(deftest lt-2-test
  (check-op-backpropagates-nil lt-2)
  (testing "scalar < scalar"
    (evaluates-to? (lt-2 1 0) 0)
    (evaluates-to? (lt-2 0 1) 1)
    (evaluates-to? (lt-2 0 0) 0)
    (evaluates-to? (lt-2 3 -2) 0)
    (evaluates-to? (lt-2 -2 3) 1))
  (testing "scalar < vector"
    (evaluates-to? (lt-2 2 [-3 2 3]) [0 0 1]))
  (testing "scalar < matrix"
    (evaluates-to? (lt-2 2 [[-3 2] [0 3]]) [[0 0] [0 1]]))
  (testing "vector < scalar"
    (evaluates-to? (lt-2 [-3 2 3] 2) [1 0 0]))
  (testing "vector < matrix"
    (evaluates-to? (lt-2 [-2 2] [[-3 2] [0 0]]) [[0 0] [1 0]]))
  (testing "matrix < scalar"
    (evaluates-to? (lt-2 [[-3 2] [0 3]] 2) [[1 0] [1 0]]))
  (testing "matrix < vector"
    (evaluates-to? (lt-2 [[-3 2] [0 3]] [-2 2]) [[1 0] [0 0]]))
  (testing "matrix < matrix"
    (evaluates-to? (lt-2 [[-3 2] [0 3]] [[-5 3.14] [0 2]]) [[0 1] [0 0]])))

(deftest lt-test
  #_(testing "0 operands"
      (is (thrown? Exception (evaluate (lt)))))
  #_(testing "1 operand"
      (evaluates-to? (lt 0) 1)
      (evaluates-to? (lt [1 2 3]) [1 1 1])
      (evaluates-to? (lt [[1 2] [3 4]]) [[1 1] [1 1]]))
  (testing "2 operands"
    (evaluates-to? (lt 3 7) 1)
    (evaluates-to? (lt 7 3) 0)
    (evaluates-to? (lt 7 7) 0)
    (evaluates-to? (lt [7 3 0] 3) [0 0 1])
    (evaluates-to? (lt 3 [7 3 0]) [1 0 0])
    (evaluates-to? (lt [[1 2] [3 4]] 3) [[1 1] [0 0]])
    (evaluates-to? (lt 3 [[1 2] [3 4]]) [[0 0] [0 1]])
    (evaluates-to? (lt [[1 2] [3 4]] [3 2]) [[1 0] [0 0]])
    (evaluates-to? (lt [3 2] [[1 2] [3 4]]) [[0 0] [0 1]])
    (evaluates-to? (lt [[1 2] [3 4]] [[1 3] [3 2]]) [[0 1] [0 0]])
    (evaluates-to? (lt [[[1 2] [3 4]] [[5 6] [7 8]]]
                       [[1 6] [3 8]])
                   [[[0 1] [0 1]] [[0 0] [0 0]]]))
  (testing "3 operands"
    (evaluates-to? (lt 3 4 5) 1)
    (evaluates-to? (lt 5 4 3) 0)
    (evaluates-to? (lt 5 3 4) 0)
    (evaluates-to? (lt 3 5 4) 0)
    (evaluates-to? (lt 5 4 4) 0)
    (evaluates-to? (lt 5 5 4) 0)
    (evaluates-to? (lt [3 5 5 4 5 5]
                       [4 4 3 5 4 5]
                       [5 3 4 4 4 4])
                   [1 0 0 0 0 0])
    (evaluates-to? (lt [3 5 6 3 5 4]
                       4
                       [5 3 5 3 4 3])
                   [1 0 0 0 0 0])
    (evaluates-to? (lt [[1.0 2.4 3.3] [4.4 5.5 6.6]]
                       [[1.0 2.3 3.3] [4.5 5.5 6.0]]
                       [[1.0 2.2 3.4] [4.6 5.5 6.0]])
                   [[0 0 0] [1 0 0]])))

(deftest le-2-test
  (check-op-backpropagates-nil le-2)
  (testing "scalar <= scalar"
    (evaluates-to? (le-2 1 0) 0)
    (evaluates-to? (le-2 0 1) 1)
    (evaluates-to? (le-2 0 0) 1)
    (evaluates-to? (le-2 3 -2) 0)
    (evaluates-to? (le-2 -2 3) 1))
  (testing "scalar <= vector"
    (evaluates-to? (le-2 2 [-3 2 3]) [0 1 1]))
  (testing "scalar <= matrix"
    (evaluates-to? (le-2 2 [[-3 2] [0 3]]) [[0 1] [0 1]]))
  (testing "vector <= scalar"
    (evaluates-to? (le-2 [-3 2 3] 2) [1 1 0]))
  (testing "vector <= matrix"
    (evaluates-to? (le-2 [-2 2] [[-3 2] [0 0]]) [[0 1] [1 0]]))
  (testing "matrix <= scalar"
    (evaluates-to? (le-2 [[-3 2] [0 3]] 2) [[1 1] [1 0]]))
  (testing "matrix <= vector"
    (evaluates-to? (le-2 [[-3 2] [0 3]] [-2 2]) [[1 1] [0 0]]))
  (testing "matrix <= matrix"
    (evaluates-to? (le-2 [[-3 2] [0 3]] [[-5 3.14] [0 2]]) [[0 1] [1 0]])))

(deftest le-test
  #_(testing "0 operands"
      (is (thrown? Exception (evaluate (le)))))
  #_(testing "1 operand"
      (evaluates-to? (le 0) 1)
      (evaluates-to? (le [1 2 3]) [1 1 1])
      (evaluates-to? (le [[1 2] [3 4]]) [[1 1] [1 1]]))
  (testing "2 operands"
    (evaluates-to? (le 3 7) 1)
    (evaluates-to? (le 7 3) 0)
    (evaluates-to? (le 7 7) 1)
    (evaluates-to? (le [7 3 0] 3) [0 1 1])
    (evaluates-to? (le 3 [7 3 0]) [1 1 0])
    (evaluates-to? (le [[1 2] [3 4]] 3) [[1 1] [1 0]])
    (evaluates-to? (le 3 [[1 2] [3 4]]) [[0 0] [1 1]])
    (evaluates-to? (le [[1 2] [3 4]] [3 2]) [[1 1] [1 0]])
    (evaluates-to? (le [3 2] [[1 2] [3 4]]) [[0 1] [1 1]])
    (evaluates-to? (le [[1 2] [3 4]] [[1 3] [3 2]]) [[1 1] [1 0]])
    (evaluates-to? (le [[[1 2] [3 4]] [[5 6] [7 8]]]
                       [[1 6] [3 8]])
                   [[[1 1] [1 1]] [[0 1] [0 1]]]))
  (testing "3 operands"
    (evaluates-to? (le 3 4 5) 1)
    (evaluates-to? (le 5 4 3) 0)
    (evaluates-to? (le 5 3 4) 0)
    (evaluates-to? (le 3 5 4) 0)
    (evaluates-to? (le 4 4 5) 1)
    (evaluates-to? (le 3 4 4) 1)
    (evaluates-to? (le 4 4 4) 1)
    (evaluates-to? (le [3 5 5 4 4 5]
                       [4 4 3 5 5 5]
                       [5 3 4 4 5 6])
                   [1 0 0 0 1 1])
    (evaluates-to? (le [3 5 6 3 3 4]
                       4
                       [5 3 5 3 4 5])
                   [1 0 0 0 1 1])
    (evaluates-to? (le [[1.0 2.4 3.3] [4.4 5.5 6.6]]
                       [[1.0 2.3 3.3] [4.5 5.5 6.0]]
                       [[1.0 2.2 3.4] [4.6 5.5 6.0]])
                   [[1 0 1] [1 1 0]])))

;; Trigonometric and hyperbolic ops
(deftest cos-test
  (let [[v [dn]] (test-op cos [[1 2] [3 4]] [[0 (/ pi 2)] [pi (* 1.5 pi)]])]
    (is (about= v [[1 0] [-1 0]]))
    (is (about= dn [[0 -2] [0 4]])))
  (let [g (G (cos :n))]
    (evaluates-to-about? g {:n [[1 10] [-10 -1]]} [[0.5403 -0.8391] [-0.8391 0.5403]])
    (numerically-validated? g {:n [[1 10] [-10 -1]]})))

(deftest sin-test
  (let [[v [dn]] (test-op sin [[1 2] [3 4]] [[0 (/ pi 2)] [pi (* 1.5 pi)]])]
    (is (about= v [[0 1] [0 -1]]))
    (is (about= dn [[1 0] [-3 0]])))
  (let [g (G (sin :n))]
    (evaluates-to-about? g {:n [[1 10] [-10 -1]]} [[0.8415 -0.5440] [0.5440 -0.8415]])
    (numerically-validated? g {:n [[1 10] [-10 -1]]})))

(deftest tan-test
  (let [[v [dn]] (test-op tan [[1 2] [3 4]] [[0 (/ pi 4)] [(* 0.75 pi) (- pi)]])]
    (is (about= v [[0 1] [-1 0]]))
    (is (about= dn [[1 4] [6 4]])))
  (let [g (G (tan :n))]
    (evaluates-to-about? g {:n [[1 10] [-10 -1]]} [[1.5574 0.6484] [-0.6484 -1.5574]])
    (numerically-validated? g {:n [[1 10] [-10 -1]]})))

(deftest acos-test
  (evaluates-to-about? (acos [1 -1]) [0 pi])
  (let [[v [dn]] (test-op acos 7 0)]
    (is (about= v (/ pi 2)))
    (is (about= dn -7)))
  (let [g (G (acos :n))]
    (evaluates-to-about? g {:n [[0.25 0.75] [-0.25 -0.75]]} [[1.3181 0.7227] [1.8234 2.4189]])
    (numerically-validated? g {:n [[0.25 0.75] [-0.25 -0.75]]})))

(deftest asin-test
  (evaluates-to-about? (asin [1 -1]) [(/ pi 2) (/ pi -2)])
  (let [[v [dn]] (test-op asin 7 0)]
    (is (about= v 0))
    (is (about= dn 7)))
  (let [g (G (asin :n))]
    (evaluates-to-about? g {:n [[0.25 0.75] [-0.25 -0.75]]} [[0.2527 0.8481] [-0.2527 -0.8481]])
    (numerically-validated? g {:n [[0.25 0.75] [-0.25 -0.75]]})))

(deftest atan-test
  (let [[v [dn]] (test-op atan [[1 2] [3 4]] [[0 1] [-1 0]])]
    (is (about= v [[0 (/ pi 4)] [(/ pi -4) 0]]))
    (is (about= dn [[1 1] [1.5 4]])))
  (let [g (G (atan :n))]
    (evaluates-to-about? g {:n [[1 10] [-10 -1]]} [[0.7853 1.4711] [-1.4711 -0.7853]])
    (numerically-validated? g {:n [[0.25 0.75] [-0.25 -0.75]]})))

(deftest cosh-test
  (let [[v [dn]] (test-op cosh [[1 2] [3 4]] [[-2 -1] [0 1]])]
    (is (about= v [[3.7620 1.54308] [1 1.5431]] 0.001))
    (is (about= dn [[-3.6268 -2.3504] [0 4.7008]] 0.001)))
  (let [g (G (cosh :n))]
    (evaluates-to-about? g {:n [[5 3] [-4 -10]]} [[74.2099 10.0677] [27.3082 11013.2329]])
    (numerically-validated? g {:n [[5 3] [-4 -10]]} 0.1)))

(deftest sinh-test
  (let [[v [dn]] (test-op sinh [[1 2] [3 4]] [[-2 -1] [0 1]])]
    (is (about= v [[-3.6268 -1.1752] [0 1.1752]] 0.001))
    (is (about= dn [[3.7620 3.0861] [3 6.1723]] 0.001)))
  (let [g (G (sinh :n))]
    (evaluates-to-about? g {:n [[5 3] [-4 -10]]} [[74.2032 10.0179] [-27.2899 -11013.2329]])
    (numerically-validated? g {:n [[5 3] [-4 -10]]} 0.1)))

(deftest tanh-test
  (let [[v [dn]] (test-op tanh [[1 2] [3 4]] [[-2 -0.5] [0.25 0.75]])]
    (is (about= v [[-0.9640 -0.4621] [0.2449 0.6351]] 0.001))
    (is (about= dn [[0.07065 1.5729] [2.8200 2.3863]] 0.001)))
  (let [g (G (tanh :n))]
    (evaluates-to-about? g {:n [[-0.1 0] [0.5 10]]} [[-0.0996 0] [0.4621 1]])
    (numerically-validated? g {:n (m/matrix [[-0.1 0] [0.5 3]])} 0.1)))

;; Propagation tests
(deftest propagate-linear-test
  (let [g (G (+ (* :a :x) :b))]
    (testing "forward"
      (evaluates-to? g {:a 3 :x 2 :b 4} 10)
      (evaluates-to? g {:a [1 2 3] :x [4 5 6] :b 8} [12 18 26])
      (evaluates-to? g {:a [[1 2] [3 4]]
                        :x [[5 6] [7 8]]
                        :b [10 20]} [[15 32] [31 52]]))
    (testing "backward"
      (numerically-validated? g {:a 3 :x 2 :b 4})
      (numerically-validated? g {:a -2 :x -7 :b 13})
      (numerically-validated? g {:a -2 :x 0 :b 13})
      (numerically-validated? g {:a [1 2 3] :x [4 5 6] :b 8})
      (numerically-validated? g {:a [[1 2] [3 4]]
                                 :x [[5 6] [7 8]]
                                 :b [10 20]}))))

(deftest propagate-linear-pow-test
  (let [g (G (pow (+ (* :a :x) :b) :pow))]
    (testing "forward"
      (evaluates-to? g {:a 3 :x 2 :b 4 :pow 2} 100)
      (evaluates-to? g {:a 3 :x 2 :b 4 :pow [0 1 2]} [1 10 100])
      (evaluates-to? g {:a [1 2 3] :x [4 5 6] :b 8 :pow 2} [144 324 676]))
    (testing "backward"
      (numerically-validated? g {:a 3 :x 2 :b 4 :pow 2})
      (numerically-validated? g {:a 2 :x 14 :b 3 :pow 0.1})
      (numerically-validated? g {:a 3 :x 2 :b 4 :pow [0 1 2]})
      (numerically-validated? g {:a [1 2 3] :x [4 5 6] :b 8 :pow 2}))))

(deftest propagate-linear-*-linear-test
  (let [linear (G (+ (* :a :x) :b))
        g (G (* linear linear))]
    (testing "forward"
      (evaluates-to? g {:a 3 :x 2 :b 4} 100)
      (evaluates-to? g {:a [1 2 3] :x [4 5 6] :b 8} [144 324 676]))
    (testing "backward"
      (numerically-validated? g {:a 3 :x 2 :b 4})
      (numerically-validated? g {:a -3 :x 2 :b -4})
      (numerically-validated? g {:a [1 2 3] :x [4 5 6] :b 8}))))

(deftest propagate-linear-broadcasting-test
  (let [g (G (add (mul :a :x) :b))]
    (testing "forward"
      (evaluates-to? g {:a [[1 2 3]
                            [4 5 6]]
                        :x [[2] [3]]
                        :b [[10 20 30]]} [[12 24 36]
                                          [22 35 48]]))
    (testing "backward"
      (numerically-validated? g {:a [[1 2 3]
                                     [4 5 6]]
                                 :x [[2] [3]]
                                 :b [[10 20 30]]}))))

(deftest mmul-propagation-test
  (let [g (G (-> (mmul :a :b)
                 (mmul :c)))]
    (testing "forward"
      (evaluates-to? g {:a [[1 2 3]
                            [4 5 6]]
                        :b [[3 4]
                            [0 1]
                            [6 9]]
                        :c [[3 5 1]
                            [7 7 4]]} [[294 336 153]
                                       [669 765 348]]))
    (testing "backward"
      (numerically-validated? g {:a [[1 2 3]
                                     [4 5 6]]
                                 :b [[3 4]
                                     [0 1]
                                     [6 9]]
                                 :c [[3 5 1]
                                     [7 7 4]]}))))

(deftest propagate-mmul-sum-add-mmul-test
  (let [g (G (mmul (add
                     (sum-along (mmul :a :b) {:axis 0
                                              :collapse false})
                     :c)
                   :d))]
    (testing "forward"
      (evaluates-to? g {:a [[1 2]
                            [3 4]]
                        :b [[0 4 9]
                            [2 5 4]]
                        :c 3
                        :d [[4 2]
                            [6 8]
                            [0 1]]} [[354 485]]))
    (testing "backward"
      (numerically-validated? g {:a [[1 2]
                                     [3 4]]
                                 :b [[0 4 9]
                                     [2 5 4]]
                                 :c 3
                                 :d [[4 2]
                                     [6 8]
                                     [0 1]]}))))

(deftest mmul-norm-repeatedly-test
  (let [repeats 10
        f (fn [input]
            (first
              (drop (dec repeats)
                    (iterate
                      (fn [nd]
                        (G (div
                             (mmul nd (transpose input))
                             (esum nd))))
                      input))))
        g (G (f :x))]
    (is (nd= (evaluate g {:x [[2 2] [2 2]]}) [[1 1] [1 1]]))
    (numerically-validated? g {:x 2})
    (numerically-validated? g {:x [1 2 3]})
    (numerically-validated? g {:x [[1 2] [3 4]]})
    ; NOTE: This is a crazy test that takes a little bit to run.
    #_(numerically-validated? g {:x [[[1 2] [3 4]] [[5 6] [7 8]]]} 1e-3)))

(deftest min-max-test
  (let [repeats 10
        [a b c d e f g h i j] (map #(m/sample-uniform [2 3] %) (range repeats))
        inputs {:a a :b b :c c :d d :e e :f f :g g :h h :i i :j j}
        g (G (max :j (min :i (max :h (min :g (max :f (min :e (max :d (min :c (max :b :a))))))))))]
    (numerically-validated? g inputs)))

(deftest abs-negate-test
  (let [g (G (-> (negate :a)
                 negate
                 abs
                 negate
                 abs))
        inputs {:a [[-1 2 -3] [4 -5 6]]}]
    (is (nd= (evaluate g inputs) [[1 2 3] [4 5 6]]))
    (numerically-validated? g inputs)))

(deftest if-control-flow-test
  (let [g (G (tensor-if :flag :a :b))
        choose-a {:flag 1 :a 3 :b 7}
        choose-b {:flag 0 :a 3 :b 7}]
    (evaluates-to? g choose-a 3)
    (gradients-are? g choose-a {:a 1})
    (evaluates-to? g choose-b 7)
    (gradients-are? g choose-b {:b 1}))
  (let [g (G (* (tensor-if :flag1 :a :b)
                (tensor-if :flag2 :c :d)))
        abcd {:a 3 :b 4 :c 5 :d 6}
        choose-ac (assoc abcd :flag1 1 :flag2 1)
        choose-ad (assoc abcd :flag1 1 :flag2 0)
        choose-bc (assoc abcd :flag1 0 :flag2 1)
        choose-bd (assoc abcd :flag1 0 :flag2 0)]
    (evaluates-to? g choose-ac 15)
    (gradients-are? g choose-ac {:a 5 :b nil :c 3 :d nil})
    (evaluates-to? g choose-ad 18)
    (gradients-are? g choose-ad {:a 6 :b nil :c nil :d 3})
    (evaluates-to? g choose-bc 20)
    (gradients-are? g choose-bc {:a nil :b 5 :c 4 :d nil})
    (evaluates-to? g choose-bd 24)
    (gradients-are? g choose-bd {:a nil :b 6 :c nil :d 4}))
  (let [g (G (+ (* 2 :a)
                (tensor-if :flag (* 3 :a) (* 4 :a))))
        flag-on {:flag 1 :a 5}
        flag-off {:flag 0 :a 5}]
    (evaluates-to? g flag-on 25)
    (gradients-are? g flag-on {:a 5})
    (evaluates-to? g flag-off 30)
    (gradients-are? g flag-off {:a 6})))
