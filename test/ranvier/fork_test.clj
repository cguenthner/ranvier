(ns ranvier.fork-test
  "Tests for `fork` op functionality in `neurojure.core`. These are in their own namespace (rather than
  in `neurojure.core-test`, because this functionality (as implemented) is very complicated and there are
  therefore a large number of tests for it."
  (:require [clojure.test :refer [deftest is testing]]
            [ranvier.core :as r :refer [G evaluate forward backward]]
            [ranvier.test-utils :refer :all]
            [tensure.core :as m]))

; Tests that inputs to `fork` are properly validated.
(deftest fork-validity-checking-test
  (let [f (G (fork [:a (slices :x)
                    :c (slices (* 2 :x))]
                   [:b :b
                    :d 0]
                   (+ :a :b (* :c :d))
                   [(* :b 2)
                    0]))
        inputs {:x [1 2 3] :b 1}]
    (testing "use of privately bound inputs out of scope throws an Exception"
      (is (thrown? Exception (evaluate (G (* f :a)) inputs)))
      (is (thrown? Exception (evaluate (G (* (+ f 1) (+ (* :a 2) 2))) inputs)))
      (is (thrown? Exception (evaluate (G (* f :c)) inputs)))
      (is (thrown? Exception (evaluate (G (* (+ f 1) (+ (* :c 2) 2))) inputs)))
      (is (thrown? Exception (evaluate (G (* (+ f 1)
                                             (+ (* (fork [:a (slices :x)]
                                                         [:f :b]
                                                         (+ :a :b)
                                                         [(* :b 2)]) 2)
                                                2))) inputs)))
      (is (thrown? Exception (evaluate (G (* (+ f 1)
                                             (+ (* (fork [:d (slices :x)]
                                                         [:f :b]
                                                         (+ :a :b)
                                                         [(* :b 2)]) 2)
                                                2))) inputs)))
      (is (thrown? Exception (evaluate (G (* f :b)) inputs)))
      (is (thrown? Exception (evaluate (G (* (+ f 1) (+ (* :b 2) 2))) inputs)))
      (is (thrown? Exception (evaluate (G (* f :d)) inputs)))
      (is (thrown? Exception (evaluate (G (* (+ f 1) (+ (* :d 2) 2))) inputs)))
      (is (thrown? Exception (evaluate (G (* (+ f 1)
                                             (+ (* (fork [:e (slices :x)]
                                                         [:c :b]
                                                         (+ :a :b)
                                                         [(* :b 2)]) 2)
                                                2))) inputs)))
      (is (thrown? Exception (evaluate (G (* (+ f 1)
                                             (+ (* (fork [:e (slices :x)]
                                                         [:b :b]
                                                         (+ :a :b)
                                                         [(* :b 2)]) 2)
                                                2))) inputs))))
    (testing "the same fork can appear in multiple parts of the graph without throwing an Exception"
      ; f evaluates to [2 4 7]
      (evaluates-to? (G (* f f)) inputs [4 16 49])
      (evaluates-to? (G (+ (* f 3)
                           (- 4 f))) inputs [8 12 18])))
  (let [make-fork-with-bindings (fn [a b c d]
                                  (G (fork [a (slices :x)
                                            b (slices (* 2 :x))]
                                           [c :b
                                            d 0]
                                           (+ :a :b (* :c :d))
                                           [(* :b 2)
                                            0])))]
    (testing "throws an exception if the same input is bound to multiple branches in the same fork"
      (is (r/node? (make-fork-with-bindings :a :b :c :d)))
      (is (thrown? Exception (make-fork-with-bindings :a :a :b :c)))
      (is (thrown? Exception (make-fork-with-bindings :a :b :b :c)))
      (is (thrown? Exception (make-fork-with-bindings :a :b :c :c)))
      (is (thrown? Exception (make-fork-with-bindings :a :b :c :a))))
    (testing "doesn't allow bindings to non-input nodes"
      (is (r/node? (make-fork-with-bindings (r/input :a) (r/input :b) (r/input :c) (r/input :d))))
      (is (thrown? Exception (make-fork-with-bindings (r/const 3) (r/input :b) (r/input :c) (r/input :d))))
      (is (thrown? Exception (make-fork-with-bindings (r/input :a) (r/const 3) (r/input :c) (r/input :d))))
      (is (thrown? Exception (make-fork-with-bindings (r/input :a) (r/input :b) (r/const 3) (r/input :d))))
      (is (thrown? Exception (make-fork-with-bindings (r/input :a) (r/input :b) (r/input :c) (r/const 3))))
      (is (thrown? Exception (make-fork-with-bindings (G (* 2 :a)) (r/input :b) (r/input :c) (r/input :d))))
      (is (thrown? Exception (make-fork-with-bindings (r/input :a) (G (* 2 :b)) (r/input :c) (r/input :d))))
      (is (thrown? Exception (make-fork-with-bindings (r/input :a) (r/input :b) (G (* 2 :c)) (r/input :d))))
      (is (thrown? Exception (make-fork-with-bindings (r/input :a) (r/input :b) (r/input :c) (G (* 2 :d)))))))
  (testing "throws an Exception if a binding form is invalid"
    (is (thrown? Exception (G (fork [:a]
                                    [:b :b]
                                    (+ :a :b)
                                    [(* :b 2)]))))
    (is (thrown? Exception (G (fork [:a (slices :x)
                                     :c]
                                    [:b :b]
                                    (+ :a :b)
                                    [(* :b 2)]))))
    (is (thrown? Exception (G (fork [:a (slices :x)]
                                    [:b]
                                    (+ :a :b)
                                    [(* :b 2)]))))
    (is (thrown? Exception (G (fork [:a (slices :x)]
                                    [:b 7 :c]
                                    (+ :a :b)
                                    [(* :b 2)]))))
    (is (thrown? Exception (G (fork {:a (slices :x)}
                                    [:b 7]
                                    (+ :a :b)
                                    [(* :b 2)]))))
    (is (thrown? Exception (G (fork [:a (slices :x)]
                                    {:b 7}
                                    (+ :a :b)
                                    [(* :b 2)])))))
  (testing "throws and Exception if number of recur branches does not equal number of recur bindings"
    (is (thrown? Exception (G (fork [:a (slices :x)]
                                    [:b :b]
                                    (+ :a :b)
                                    [(* :b 2)
                                     (* 3 :b)]))))
    (is (thrown? Exception (G (fork [:a (slices :x)]
                                    [:b :b
                                     :c 0]
                                    (+ :a :b)
                                    [(* :b 2)]))))))

; Checks that `fork` works with the different types of inputs and branches it can receive.
(deftest fork-input-test
  (testing "works with different types of input nodes"
    (let [g (G (fork [:a (slices :x)]
                     [:b :b]
                     (* :a :b)
                     [(* :b 2)]))
          inputs {:x [3 5 7] :b 4}]
      (evaluates-to? g inputs [12 40 112])
      (gradients-are? g inputs {:x [4 8 16]
                                :b 41}))
    (let [g (G (fork [:a (slices (+ :x :y))]
                     [:b :b]
                     (* :a :b)
                     [(* :b 2)]))
          inputs {:x [1 2 3] :y [2 3 4] :b 4}]
      (evaluates-to? g inputs [12 40 112])
      (gradients-are? g inputs {:x [4 8 16]
                                :y [4 8 16]
                                :b 41}))
    (let [x (r/const (map m/array [3 5 7]) :x)
          g (G (named :g (fork [:a x]
                               [:b :b]
                               (* :a :b)
                               [(* :b 2)])))
          vals (forward g {:b 4})
          {x-grad :x
           b-grad :b} (backward g vals [:x :b])]
      (is (= (get vals (r/get-node-name g)) (m/array [12 40 112])))
      (is (= x-grad (map m/array [4 8 16])))
      (is (m/equals b-grad (m/array 41))))
    (let [g (G (named :g (fork [:a :x]
                               [:b :b]
                               (* :a :b)
                               [(* :b 2)])))
          inputs {:x (map m/array [3 5 7]) :b 4}
          vals (forward g inputs)
          {x-grad :x
           b-grad :b} (backward g vals [:b :x])]
      (is (= (:g vals) (m/array [12 40 112])))
      (is (= x-grad (map m/array [4 8 16])))
      (is (= b-grad (m/array 41))))
    (let [g (G (named :g (fork [:a (r/input :x (fn [_] (map m/array [3 5 7])))]
                               [:b :b]
                               (* :a :b)
                               [(* :b 2)])))
          inputs {:x (map m/array [3 5 7]) :b 4}
          vals (forward g inputs)
          {x-grad :x
           b-grad :b} (backward g vals [:b :x])]
      (is (= (:g vals) (m/array [12 40 112])))
      (is (= x-grad (map m/array [4 8 16])))
      (is (= b-grad (m/array 41)))))
  (testing "works with different types of recur initializer nodes"
    (let [b-start (r/const (m/array 4) :b-start)
          g (G (fork [:a (slices :x)]
                     [:b b-start]
                     (* :a :b)
                     [(* :b 2)]))
          inputs {:x [3 5 7]}]
      (evaluates-to? g inputs [12 40 112])
      (gradients-are? g inputs {:x [4 8 16]
                                :b-start 41}))
    (let [g (G (fork [:a (slices :x)]
                     [:b (- (* 7 :c) 17)]
                     (* :a :b)
                     [(* :b 2)]))
          inputs {:x [3 5 7] :c 3}]
      (evaluates-to? g inputs [12 40 112])
      (gradients-are? g inputs {:x [4 8 16]
                                :c 287}))
    (let [g (G (fork [:a (slices :x)]
                     [:b (r/input :b-start (fn [_] (m/array 4)))]
                     (* :a :b)
                     [(* :b 2)]))
          inputs {:x [3 5 7]}]
      (evaluates-to? g inputs [12 40 112])
      (gradients-are? g inputs {:x [4 8 16]
                                :b-start 41})))
  (testing "works with different types of recurring branches"
    (let [g (G (fork [:a (slices :x)]
                     [:b :b]
                     (* :a :b)
                     [2]))
          inputs {:x [3 5 7] :b 4}]
      (evaluates-to? g inputs [12 10 14])
      (gradients-are? g inputs {:x [4 2 2]
                                :b 3}))
    (let [g (G (fork [:a (slices :x)]
                     [:b :b]
                     (* :a :b)
                     [:c]))
          inputs {:x [3 5 7] :b 4 :c 2}]
      (evaluates-to? g inputs [12 10 14])
      (gradients-are? g inputs {:x [4 2 2]
                                :b 3
                                :c 12}))
    (let [g (G (fork [:a (slices :x)]
                     [:b :b]
                     (* :a :b)
                     [(r/input :c (fn [_] (m/array 2)))]))
          inputs {:x [3 5 7] :b 4}]
      (evaluates-to? g inputs [12 10 14])
      (gradients-are? g inputs {:x [4 2 2]
                                :b 3
                                :c 12})))
  (testing "works different types of output branches"
    (let [g (G (fork [:a (slices :x)]
                     [:b :b]
                     7
                     [(* :b 2)]))
          inputs {:x [3 5 7] :b 4}]
      (evaluates-to? g inputs [7 7 7])
      (gradients-are? g inputs {:x [0 0 0]
                                :b 0}))
    (let [g (G (fork [:a (slices :x)]
                     [:b :b]
                     (r/const (m/array 7) :c)
                     [(* :b 2)]))
          inputs {:x [3 5 7] :b 4}]
      (evaluates-to? g inputs [7 7 7])
      (gradients-are? g inputs {:x [0 0 0]
                                :b 0
                                :c 3}))
    (let [g (G (fork [:a (slices :x)]
                     [:b :b]
                     :c
                     [(* :b 2)]))
          inputs {:x [3 5 7] :b 4 :c 7}]
      (evaluates-to? g inputs [7 7 7])
      (gradients-are? g inputs {:x [0 0 0]
                                :b 0
                                :c 3}))
    (let [g (G (fork [:a (slices :x)]
                     [:b :b]
                     (r/input :c (fn [_] (m/array 7)))
                     [(* :b 2)]))
          inputs {:x [3 5 7] :b 4 :c 7}]
      (evaluates-to? g inputs [7 7 7])
      (gradients-are? g inputs {:x [0 0 0]
                                :b 0})))
  (testing "works with different types of `output-time-axis` nodes"
    (let [make-graph (fn [output-time-axis]
                       (G (fork [:a (slices :x 2)
                                 :b (slices :y 0)]
                                [:c :c]
                                (* :a :b :c)
                                [(* :c 2)]
                                {:output-time-axis output-time-axis})))
          inputs {:x [[[1 2 3] [4 5 6]]
                      [[7 8 9] [10 11 12]]] :y [0 1 2] :c 1}
          axis1-result [[[0 0] [4 10] [24 48]]
                        [[0 0] [16 22] [72 96]]]
          axis2-result [[[0 4 24] [0 10 48]]
                        [[0 16 72] [0 22 96]]]]
      (let [g1 (make-graph 1)
            g2 (make-graph 2)]
        (evaluates-to? g1 inputs axis1-result)
        (numerically-validated? g1 inputs)
        (evaluates-to? g2 inputs axis2-result)
        (numerically-validated? g2 inputs))
      (let [g1 (make-graph (r/const 1))
            g2 (make-graph (r/const 2))]
        (evaluates-to? g1 inputs axis1-result)
        (numerically-validated? g1 inputs)
        (evaluates-to? g2 inputs axis2-result)
        (numerically-validated? g2 inputs))
      (let [g1 (make-graph :d)
            g2 (make-graph :d)
            g1-inputs (assoc inputs :d 1)
            g2-inputs (assoc inputs :d 2)]
        (evaluates-to? g1 g1-inputs axis1-result)
        (numerically-validated? g1 g1-inputs)
        (evaluates-to? g2 g2-inputs axis2-result)
        (numerically-validated? g2 g2-inputs))
      (let [g1 (make-graph (r/input :d (fn [_] (m/array 1))))
            g2 (make-graph (r/input :d (fn [_] (m/array 2))))]
        (evaluates-to? g1 inputs axis1-result)
        (numerically-validated? g1 inputs)
        (evaluates-to? g2 inputs axis2-result)
        (numerically-validated? g2 inputs)))))

; Tests that `fork` works with different basic input / recurrence architectures.
(deftest fork-architecture-test
  (testing "common usage with single recur branch"
    (let [g (G (fork [:a (slices :x)]
                     [:b :b]
                     (+ :a :b)
                     [(* :b 2)]))
          inputs {:x [3 5 7] :b 4}]
      (evaluates-to? g inputs [7 13 23])
      (gradients-are? g inputs {:x [1 1 1]
                                :b 7}))
    (let [g (G (fork [:a (slices :x)]
                     [:b :b]
                     (* :a :b)
                     [(* :b 2)]))
          inputs {:x [3 5 7] :b 4}]
      (evaluates-to? g inputs [12 40 112])
      (gradients-are? g inputs {:x [4 8 16]
                                :b 41}))
    (let [g (G (fork [:a (slices :x)]
                     [:b :b]
                     (* :a :b)
                     [(* (pow :b 2) 2)]))
          inputs {:x [3 5 7] :b 4}]
      (evaluates-to? g inputs [12 160 14336])
      (gradients-are? g inputs {:x [4 32 2048]
                                :b 14419})))
  (testing "extra input (:c) in recurred graph"
    (let [g (G (fork [:a (slices :x)]
                     [:b :b]
                     (* :a :b)
                     [(* :b 2 :c)]))
          inputs {:x [3 5 7] :b 4 :c 3}]
      (evaluates-to? g inputs [12 120 1008])
      (gradients-are? g inputs {:x [4 24 144]
                                :b 285
                                :c 712})))
  (testing "extra input (:c) in both recurred branch and output branch"
    (let [g (G (fork [:a (slices :x)]
                     [:b :b]
                     (* :a :b :c)
                     [(* :b 2 :c)]))
          inputs {:x [3 5 7] :b 4 :c 3}]
      (evaluates-to? g inputs [36 360 3024])
      (gradients-are? g inputs {:x [12 72 432]
                                :b 855
                                :c 3276})))
  (testing "forked-over input (:a) is used in both an output and recurred-branch"
    (let [g (G (fork [:a (slices :x)]
                     [:b :b]
                     (* :a :b)
                     [(* :a :b 2)]))
          inputs {:x [3 5 7] :b 4 :c 3}]
      (evaluates-to? g inputs [12 120 1680])
      (numerically-validated? g inputs)))
  (testing "forked-over input (:a) is used only in the recurred branch"
    (let [g (G (fork [:a (slices :x)]
                     [:b :b]
                     (* :b 2)
                     [(* :a :b 2)]))
          inputs {:x [3 5 7] :b 4}]
      (evaluates-to? g inputs [8 48 480])
      (numerically-validated? g inputs)))
  (testing "forked-over input (:a) is also a recur target"
    (let [g (G (fork
                 [:a (slices :x)]
                 [:c :a]
                 (* :b 2)
                 [(* :c 2)]))
          inputs {:x [3 5 7] :b 4}]
      (evaluates-to? g inputs [8 8 8])
      (numerically-validated? g inputs)))
  (testing "standard usage with two recur branches"
    (let [g (G (fork [:a (slices :x)]
                     [:b :b
                      :c :c]
                     (* :a :b :c)
                     [(* :a :b 2)
                      (+ :c 1)]))
          inputs {:x [3 5 7] :b 4 :c 3}]
      (evaluates-to? g inputs [36 480 8400])
      (numerically-validated? g inputs)))
  (testing "recur target of one branch (:c) is used in another recur branch"
    (let [g (G (fork [:a (slices :x)]
                     [:b :b
                      :c :c]
                     (* :a :b :c)
                     [(* :a :b 2)
                      (+ :c :b 1)]))
          inputs {:x [3 5 7] :b 4 :c 3}]
      (evaluates-to? g inputs [36 960 55440])
      (numerically-validated? g inputs)))
  (testing "recur target for both recur branches (:b and :c) are used in each other"
    (let [g (G (fork [:a (slices :x)]
                     [:b :b
                      :c :c]
                     (* :a :b :c)
                     [(* :a :b :c)
                      (+ :c :b 1)]))
          inputs {:x [3 5 7] :b 4 :c 3}]
      (evaluates-to? g inputs [36 1440 453600])
      (numerically-validated? g inputs)))
  (testing "recur branch that is not used in the output branch (i.e. a dead recur branch)"
    (let [g (G (fork [:a (slices :x)]
                     [:b :b
                      :c :c]
                     (* :a :b)
                     [(* :a :b 2)
                      (+ :c 1)]))
          inputs {:x [3 5 7] :b 4 :c 3}]
      (evaluates-to? g inputs [12 120 1680])
      (numerically-validated? g inputs)))
  (testing "no recur branches"
    (let [g (G (fork [:a (slices :x)]
                     []
                     (* :a :b :c)
                     []))
          inputs {:x [3 5 7] :b 4 :c 3}]
      (evaluates-to? g inputs [36 60 84])
      (numerically-validated? g inputs)))
  (testing "recur branch does not utilize the target (:c) in its calculation"
    (let [g (G (fork
                 [:a (slices :x)]
                 [:c :a]
                 (* :c 2)
                 [(* :b 2)]))
          inputs {:x [3 5 7] :b 4}]
      (evaluates-to? g inputs [6 16 16])
      (numerically-validated? g inputs)))
  (testing "two recur branch do not utilize their own targets in their calculations but depend on different values"
    (let [g (G (fork [:a (slices :x)]
                     [:d :a
                      :c :c]
                     (* :d :c 2)
                     [(* :b 2)
                      (* :b 3)]))
          inputs {:x [3 5 7] :b 4 :c 3}]
      (evaluates-to? g inputs [18 192 192])
      (numerically-validated? g inputs)))
  (testing "two recur branch do not utilize their own targets in their calculations but utilize each other's values"
    (let [g (G (fork [:a (slices :x)]
                     [:b :b
                      :c :c]
                     (* :a :b :c 2)
                     [(* :c 2)
                      (+ :b 2)]))
          inputs {:x [3 5 7] :b 4 :c 3}]
      (evaluates-to? g inputs [72 360 1344])
      (numerically-validated? g inputs)))
  (testing "recur branch does not utilize the target (:a) in its calculation, which is also not utilized at all"
    (let [g (G (fork [:a (slices :x)]
                     [:d :a]
                     (* :c 2)
                     [(* :b 2)]))
          inputs {:x [3 5 7] :b 4 :c 3}]
      (evaluates-to? g inputs [6 6 6])
      (numerically-validated? g inputs)))
  (testing "the same sub-branch is used in the output and a recurred branch of a fork"
    (let [subbranch (G (+ (* :d :b) :c))
          g (G (fork [:a (slices :x)]
                     [:d :a]
                     subbranch
                     [subbranch]))
          inputs {:x [1 2 3] :b 2 :c 3}]
      (evaluates-to? g inputs [5 13 29])
      (numerically-validated? g inputs))
    (let [subbranch (G (+ (* :a :b) :c))
          g (G (fork
                 [:a (slices :x)]
                 [:b :b]
                 subbranch
                 [subbranch]))
          inputs {:x [1 2 3] :b 2 :c 3}]
      (evaluates-to? g inputs [5 13 42])
      (numerically-validated? g inputs)))
  (testing "the same sub-branch is used in two recurred branches and the output"
    (let [subbranch (G (+ (* :d :b) :c))
          g (G (fork
                 [:a (slices :x)]
                 [:d :a
                  :c :c]
                 subbranch
                 [subbranch
                  (+ subbranch 1)]))
          inputs {:x [1 2 3] :b 2 :c 3}]
      (evaluates-to? g inputs [5 16 49])
      (numerically-validated? g inputs))
    (let [subbranch (G (+ (* :d :b) :c))
          g (G (fork
                 [:a (slices :x)]
                 [:d :a
                  :b :b]
                 subbranch
                 [subbranch
                  (+ subbranch 1)]))
          inputs {:x [1 2 3] :b 2 :c 3}]
      (evaluates-to? g inputs [5 33 1125])
      (numerically-validated? g inputs)))
  (testing "works with inputs needing initialization"
    ; Node needing initialization is in a recur branch
    (let [b (r/input :b (fn [_] (m/array 1)))
          g (G (fork [:a (slices :x)]
                     [:c :c]
                     (* :a :c)
                     [(* b 2)]))
          inputs {:x [3 5 7] :c 1}]
      (evaluates-to? g inputs [3 10 14])
      (gradients-are? g inputs {:x [1 2 2]
                                :b 24}))
    (let [b (r/input :b (fn [_] (m/array 1)))
          g (G (fork [:a (slices :x)]
                     [b b]
                     (* :a b)
                     [(* b 2)]))
          inputs {:x [3 5 7]}]
      (evaluates-to? g inputs [3 10 28])
      (gradients-are? g inputs {:x [1 2 4]
                                :b 41}))
    ; Node needing initialization is in an output branch
    (let [c (r/input :c (fn [_] (m/array 3)))
          g (G (fork [:a (slices :x)]
                     [:b :b]
                     (* :a :b c)
                     [(* :b 2)]))
          inputs {:x [3 5 7] :b 1}]
      (evaluates-to? g inputs [9 30 84])
      (gradients-are? g inputs {:x [3 6 12]
                                :b 123
                                :c 41}))
    ; Node needing initialization is forked over
    (let [x (r/input :x (fn [_] (m/array [3 5 7])))
          g (G (fork [:a (slices x)]
                     [:b :b]
                     (* :a :b)
                     [(* :b 2)]))
          inputs {:b 1}]
      (evaluates-to? g inputs [3 10 28])
      (gradients-are? g inputs {:x [1 2 4]
                                :b 41}))
    ; All of the above
    (let [x (r/input :x (fn [_] (m/array [3 5 7])))
          b (r/input :b (fn [_] (m/array 1)))
          c (r/input :c (fn [_] (m/array 3)))
          g (G (fork [:a (slices x)]
                     [b b]
                     (* :a b c)
                     [(* b 2)]))]
      (evaluates-to? g {} [9 30 84])
      (gradients-are? g {} {:x [3 6 12] :b 123 :c 41})))
  (testing "a subranch is its own recur target"
    (let [subbranch (G (+ (* :a :b) :c))
          g (G (fork [:a (slices :x)]
                     [:e subbranch]
                     (* :e :a)
                     [(* :e :d)]))
          inputs {:x [1 2 3] :b 2 :c 3 :d 2}]
      (evaluates-to? g inputs [5 20 60])
      (numerically-validated? g inputs))
    (let [sb1 (G (+ (* :a :b) :c))
          sb2 (G (* (+ :a :c) :b))
          g (G (fork [:a (slices :x)]
                     [:sb1 sb1
                      :sb2 sb2]
                     (* :sb1 :sb2)
                     [(* :sb1 :d)
                      (* :sb2 :e)]))
          inputs {:x [1 2 3] :b 2 :c 3 :d 2 :e 3}]
      (evaluates-to? g inputs [40 240 1440])
      (numerically-validated? g inputs)))
  (testing "branches have subbranches that are constant"
    (let [subbranch (G (+ (* :a :b) (* :c :d)))
          g (G (fork [:a (slices :x)]
                     [:sb subbranch]
                     (* :sb :a (* :d :e))
                     [(* :sb (* :d :e))]))
          inputs {:x [1 2 3] :b 2 :c 1.5 :d 2 :e 1}]
      (evaluates-to? g inputs [10 40 120])
      (numerically-validated? g inputs))
    (let [sb1 (G (+ (* :a (* :b :f)) (* :c :d)))
          sb2 (G (* (+ :a (* :c :d)) (* :b :f)))
          g (G (fork [:a (slices :x)]
                     [:sb1 sb1
                      :sb2 sb2]
                     (* :sb1 :sb2)
                     [(* :sb1 (- :e :f))
                      (* :sb2 (/ (+ (* :c :d) :e) :b))]))
          inputs {:x [1 2 3] :b 2 :c 1.5 :d 2 :e 3 :f 1}]
      (evaluates-to? g inputs [40 240 1440])
      (numerically-validated? g inputs))))

(deftest fork-input-size-test
  (testing "works with matrices forked-over different axes"
    (let [g (G (fork [:a (slices :x)]
                     [:b :b]
                     (* :a :b)
                     [(* :b 2)]))
          inputs {:x [[1 2 3]
                      [4 5 6]] :b 2}]
      (evaluates-to? g inputs [[2 4 6]
                               [16 20 24]])
      (numerically-validated? g inputs))
    (let [g (G (fork [:a (slices :x)]
                     [:b :b]
                     (* :a :b)
                     [(* :b 2)] {:output-time-axis 1}))
          inputs {:x [[1 2 3]
                      [4 5 6]] :b 2}]
      (evaluates-to? g inputs [[2 16]
                               [4 20]
                               [6 24]])
      (numerically-validated? g inputs))
    (let [g (G (fork [:a (slices :x 1)]
                     [:b :b]
                     (* :a :b)
                     [(* :b 2)] {:output-time-axis 1}))
          inputs {:x [[1 2 3]
                      [4 5 6]] :b 2}]
      (evaluates-to? g inputs [[2 8 24]
                               [8 20 48]])
      (numerically-validated? g inputs))
    (let [g (G (fork [:a (slices :x 1)]
                     [:b :b]
                     (* :a :b)
                     [(* :b 2)]))
          inputs {:x [[1 2 3]
                      [4 5 6]] :b 2}]
      (evaluates-to? g inputs [[2 8]
                               [8 20]
                               [24 48]])
      (numerically-validated? g inputs)))
  (testing "works when forking over a matrix and a vector without recurrence"
    (let [g (G (fork
                 [:a (slices :x)
                  :b (slices :y)]
                 []
                 (* :a :b)
                 []))
          inputs {:x [[1 2 3]
                      [4 5 6]] :y [7 8]}]
      (evaluates-to? g inputs [[7 14 21]
                               [32 40 48]])
      (numerically-validated? g inputs))
    (let [g (G (fork [:a (slices :x)
                      :b (slices :y)]
                     []
                     (* :a :b)
                     []
                     {:output-time-axis 1}))
          inputs {:x [[1 2 3]
                      [4 5 6]] :y [7 8]}]
      (evaluates-to? g inputs [[7 32]
                               [14 40]
                               [21 48]])
      (numerically-validated? g inputs))
    (let [g (G (fork [:a (slices :x 1)
                      :b (slices :y 0)]
                     []
                     (* :a :b)
                     []))
          inputs {:x [[1 2 3]
                      [4 5 6]] :y [7 8 9]}]
      (evaluates-to? g inputs [[7 28]
                               [16 40]
                               [27 54]])
      (numerically-validated? g inputs))
    (let [g (G (fork [:a (slices :x 1)
                      :b (slices :y 0)]
                     []
                     (* :a :b)
                     []
                     {:output-time-axis 1}))
          inputs {:x [[1 2 3]
                      [4 5 6]] :y [7 8 9]}]
      (evaluates-to? g inputs [[7 16 27]
                               [28 40 54]])
      (numerically-validated? g inputs)))
  (testing "works when forking over a matrix and a vector with simple recurrence"
    (let [g (G (fork [:a (slices :x 1)
                      :b (slices :y 0)]
                     [:c :c
                      :d :d]
                     (* :a :b :c :d)
                     [(* :c 2)
                      (* :d 3)]
                     {:output-time-axis 1}))
          inputs {:x [[1 2 3]
                      [4 5 6]] :y [7 8 9] :c 1 :d 2}]
      (evaluates-to? g inputs [[14 192 1944]
                               [56 480 3888]])
      (numerically-validated? g inputs 0.1))
    (let [g (G (fork [:a (slices :x 1)
                      :b (slices :y 0)]
                     [:c :c
                      :d :d]
                     (* :a :b :c :d)
                     [(* :c 2)
                      (* :d 3)]))
          inputs {:x [[1 2 3]
                      [4 5 6]] :y [7 8 9] :c 1 :d 2}]
      (evaluates-to? g inputs [[14 56]
                               [192 480]
                               [1944 3888]])
      (numerically-validated? g inputs 0.1))
    (let [g (G (fork [:a (slices :x)
                      :b (slices :y)]
                     [:c :c
                      :d :d]
                     (* :a :b :c :d)
                     [(* :c 2)
                      (* :d 3)]))
          inputs {:x [[1 2 3]
                      [4 5 6]] :y [7 8] :c 1 :d 2}]
      (evaluates-to? g inputs [[14 28 42]
                               [384 480 576]])
      (numerically-validated? g inputs 0.1))
    (let [g (G (fork [:a (slices :x)
                      :b (slices :y)]
                     [:c :c
                      :d :d]
                     (* :a :b :c :d)
                     [(* :c 2)
                      (* :d 3)]
                     {:output-time-axis 1}))
          inputs {:x [[1 2 3]
                      [4 5 6]] :y [7 8] :c 1 :d 2}]
      (evaluates-to? g inputs [[14 384]
                               [28 480]
                               [42 576]])
      (numerically-validated? g inputs 0.1)))
  (testing "works when forking over a matrix and a vector with different recurrence patterns"
    (let [g (G (fork [:a (slices :x 1)
                      :b (slices :y 0)]
                     [:c :c
                      :d :d]
                     (* :c :d)
                     [(* :c :a)
                      (* :d :b)]
                     {:output-time-axis 1}))
          inputs {:x [[1 2 3]
                      [4 5 6]] :y [7 8 9] :c [1 2] :d 3}]
      (evaluates-to? g inputs [[3 21 336]
                               [6 168 6720]])
      (numerically-validated? g inputs))
    (let [g (G (fork [:a (slices :x 1)
                      :b (slices :y 0)]
                     [:c :c
                      :d :d]
                     (* :a :b :c :d)
                     [(* :c :a)
                      (* :d :b)]
                     {:output-time-axis 1}))
          inputs {:x [[1 2 3]
                      [4 5 6]] :y [7 8 9] :c [1 2] :d 3}]
      (evaluates-to? g inputs [[21 336 9072]
                               [168 6720 362880]])
      (numerically-validated? g inputs))
    (let [g (G (fork [:a (slices :x 1)
                      :b (slices :y 0)]
                     [:c :a
                      :d :b]
                     (* :c :d)
                     [(* :c 3)
                      (* :d 2)]
                     {:output-time-axis 1}))
          inputs {:x [[1 2 3]
                      [4 5 6]] :y [7 8 9]}]
      (evaluates-to? g inputs [[7 42 252]
                               [28 168 1008]])
      (numerically-validated? g inputs))
    (let [g (G (fork [:a (slices :x 1)
                      :b (slices :y 0)]
                     [:d :d
                      :c :c]
                     (* :a :d :b)
                     [(* :d 2)
                      (* :c 3)]
                     {:output-time-axis 1}))
          inputs {:x [[1 2 3]
                      [4 5 6]] :y [7 8 9] :c [10 11 12] :d 3}]
      (evaluates-to? g inputs [[21 96 324]
                               [84 240 648]])
      (numerically-validated? g inputs)))
  (testing "works when forking over two vectors"
    (let [g (G (fork [:a (slices :x)
                      :b (slices :y)]
                     [:c :a
                      :d :b]
                     (* :c :d)
                     [(* :c 3)
                      (* :d 2)]))
          inputs {:x [1 2 3] :y [4 5 6]}]
      (evaluates-to? g inputs [4 24 144])
      (numerically-validated? g inputs)))
  (testing "works when forking over a vector and a tensor"
    ; Fork over axis 2 of ;a and join-along different axes
    (let [g (G (fork [:a (slices :x 2)
                      :b (slices :y 0)]
                     [:c :c]
                     (* :a :b :c)
                     [(* :c 2)]
                     {:output-time-axis 2}))
          inputs {:x [[[1 2 3] [4 5 6]]
                      [[7 8 9] [10 11 12]]] :y [0 1 2] :c 1}]
      (evaluates-to? g inputs [[[0 4 24] [0 10 48]]
                               [[0 16 72] [0 22 96]]])
      (numerically-validated? g inputs))
    (let [g (G (fork [:a (slices :x 2)
                      :b (slices :y 0)]
                     [:c :c]
                     (* :a :b :c)
                     [(* :c 2)]
                     {:output-time-axis 1}))
          inputs {:x [[[1 2 3] [4 5 6]]
                      [[7 8 9] [10 11 12]]] :y [0 1 2] :c 1}]
      (evaluates-to? g inputs [[[0 0] [4 10] [24 48]]
                               [[0 0] [16 22] [72 96]]])
      (numerically-validated? g inputs))
    (let [g (G (fork [:a (slices :x 2)
                      :b (slices :y 0)]
                     [:c :c]
                     (* :a :b :c)
                     [(* :c 2)]))
          inputs {:x [[[1 2 3] [4 5 6]]
                      [[7 8 9] [10 11 12]]] :y [0 1 2] :c 1}]
      (evaluates-to? g inputs [[[0 0] [0 0]]
                               [[4 10] [16 22]]
                               [[24 48] [72 96]]])
      (numerically-validated? g inputs))
    ; Fork over axis 1 of ;a and join-along different axes
    (let [g (G (fork [:a (slices :x 1)
                      :b (slices :y 0)]
                     [:c :c]
                     (* :a :b :c)
                     [(* :c 2)]
                     {:output-time-axis 1}))
          inputs {:x [[[1 2 3] [4 5 6]]
                      [[7 8 9] [10 11 12]]] :y [3 4] :c 1}]
      (evaluates-to? g inputs [[[3 6 9] [32 40 48]]
                               [[21 24 27] [80 88 96]]])
      (numerically-validated? g inputs))
    (let [g (G (fork [:a (slices :x 1)
                      :b (slices :y 0)]
                     [:c :c]
                     (* :a :b :c)
                     [(* :c 2)]
                     {:output-time-axis 0}))
          inputs {:x [[[1 2 3] [4 5 6]]
                      [[7 8 9] [10 11 12]]] :y [3 4] :c 1}]
      (evaluates-to? g inputs [[[3 6 9] [21 24 27]]
                               [[32 40 48] [80 88 96]]])
      (numerically-validated? g inputs))
    (let [g (G (fork [:a (slices :x 1)
                      :b (slices :y 0)]
                     [:c :c]
                     (* :a :b :c)
                     [(* :c 2)]
                     {:output-time-axis 2}))
          inputs {:x [[[1 2 3] [4 5 6]]
                      [[7 8 9] [10 11 12]]] :y [3 4] :c 1}]
      (evaluates-to? g inputs [[[3 32] [6 40] [9 48]]
                               [[21 80] [24 88] [27 96]]])
      (numerically-validated? g inputs))
    ; Fork over axis 1 of ;a and join-along different axes
    (let [g (G (fork [:a (slices :x 0)
                      :b (slices :y 0)]
                     [:c :c]
                     (* :a :b :c)
                     [(* :c 2)]))
          inputs {:x [[[1 2 3] [4 5 6]]
                      [[7 8 9] [10 11 12]]] :y [3 4] :c 1}]
      (evaluates-to? g inputs [[[3 6 9] [12 15 18]]
                               [[56 64 72] [80 88 96]]])
      (numerically-validated? g inputs))
    (let [g (G (fork [:a (slices :x 0)
                      :b (slices :y 0)]
                     [:c :c]
                     (* :a :b :c)
                     [(* :c 2)]
                     {:output-time-axis 1}))
          inputs {:x [[[1 2 3] [4 5 6]]
                      [[7 8 9] [10 11 12]]] :y [3 4] :c 1}]
      (evaluates-to? g inputs [[[3 6 9] [56 64 72]]
                               [[12 15 18] [80 88 96]]])
      (numerically-validated? g inputs))
    (let [g (G (fork [:a (slices :x 0)
                      :b (slices :y 0)]
                     [:c :c]
                     (* :a :b :c)
                     [(* :c 2)]
                     {:output-time-axis 2}))
          inputs {:x [[[1 2 3] [4 5 6]]
                      [[7 8 9] [10 11 12]]] :y [3 4] :c 1}]
      (evaluates-to? g inputs [[[3 56] [6 64] [9 72]]
                               [[12 80] [15 88] [18 96]]])
      (numerically-validated? g inputs)))
  (testing "works when forking over two matrices"
    (let [g (G (fork [:a (slices :x 0)
                      :b (slices :y 0)]
                     [:c :c]
                     (* :a :b :c)
                     [(* :c 2)]))
          inputs {:x [[1 2 3] [4 5 6]]
                  :y [[7 8 9] [10 11 12]] :c 1}]
      (evaluates-to? g inputs [[7 16 27] [80 110 144]])
      (numerically-validated? g inputs))
    (let [g (G (fork [:a (slices :x 0)
                      :b (slices :y 1)]
                     [:c :c]
                     (* :a :b :c)
                     [(* :c 2)]))
          inputs {:x [[1 2 3] [4 5 6]]
                  :y [[7 8] [9 10] [11 12]] :c 1}]
      (evaluates-to? g inputs [[7 18 33] [64 100 144]])
      (numerically-validated? g inputs))
    (let [g (G (fork [:a (slices :x 0)
                      :b (slices :y 1)]
                     [:c :a
                      :d :b]
                     (* :c :d)
                     [(* :d 2)
                      (* :c 3)]))
          inputs {:x [[1 2 3] [4 5 6]]
                  :y [[7 8] [9 10] [11 12]]}]
      (evaluates-to? g inputs [[7 18 33] [42 108 198]])
      (numerically-validated? g inputs)))
  (testing "works when forking over a tensor and a matrix"
    (let [g (G (fork [:a (slices :x)
                      :b (slices :y)]
                     [:c :c]
                     (* :a :b :c)
                     [(* :c 2)]))
          inputs {:x [[[1 2 3] [4 5 6]]
                      [[7 8 9] [10 11 12]]]
                  :y [[2 3 0] [1 4 5]] :c [3 2 1]}]
      (evaluates-to? g inputs [[[6 12 0] [24 30 0]]
                               [[42 128 90] [60 176 120]]])
      (numerically-validated? g inputs))
    (let [g (G (fork [:a (slices :x 0)
                      :b (slices :y 1)]
                     [:c :c]
                     (* :a :b :c)
                     [(* :c 2)]))
          inputs {:x [[[1 2 3] [4 5 6]]
                      [[7 8 9] [10 11 12]]]
                  :y [[2 3] [1 4]] :c [3 2]}]
      (evaluates-to? g inputs [[[6 12 18] [8 10 12]]
                               [[126 144 162] [160 176 192]]])
      (numerically-validated? g inputs 0.1))))

; Tests that inputs needing initialization in forks are not repeatedly initialized
(deftest fork-reinitialization-test
  (testing "inputs ocurring in multiple input branches are only initialized once"
    (let [init-count (atom 0)
          h (r/input :h (fn [_]
                          (do (swap! init-count inc)
                              (m/array 1))))
          sb1 (G (+ (* (/ (+ :a :i) 2) (* :b :f)) (* :c :d)))
          sb2 (G (* (+ (/ (+ :a :i) 2) (* :c :d)) (* :b :f)))
          g (G (fork [:a (slices (+ (* :x h) h -1))
                      :i (slices (* :x (* h h)))]
                     [:sb1 sb1
                      :sb2 sb2]
                     (* :sb1 :sb2)
                     [(* :sb1 (- :e :f))
                      (* :sb2 (/ (+ (* :c :d) :e) :b))]))
          inputs {:x [1 2 3] :b 2 :c 1.5 :d 2 :e 3 :f 1}]
      (evaluates-to? g inputs [40 240 1440])
      (is (= @init-count 1))
      (numerically-validated? g inputs)))
  (testing "inputs present in multiple recurrence initialization branches are only initialized once"
    (let [init-count (atom 0)
          h (r/input :h (fn [_]
                          (do (swap! init-count inc)
                              (m/array 1))))
          sb1 (G (+ (* (/ (+ :a (* :i h)) 2) (* (* :b h) :f)) (* :c :d)))
          sb2 (G (* (+ (/ (+ :a (* :i h)) 2) (* :c :d)) (* :b (* :f h))))
          g (G (fork [:a (slices :x)
                      :i (slices :x)]
                     [:sb1 sb1
                      :sb2 sb2]
                     (* :sb1 :sb2)
                     [(* :sb1 (- :e :f))
                      (* :sb2 (/ (+ (* :c :d) :e) :b))]))
          inputs {:x [1 2 3] :b 2 :c 1.5 :d 2 :e 3 :f 1}]
      (evaluate g inputs)
      (is (= @init-count 1))))
  (testing "inputs ocurring multiple times in the output branch are only initialized once"
    (let [init-count (atom 0)
          h (r/input :h (fn [_]
                          (do (swap! init-count inc)
                              (m/array 1))))
          sb1 (G (+ (* (/ (+ :a :i) 2) (* :b :f)) (* :c :d)))
          sb2 (G (* (+ (/ (+ :a :i) 2) (* :c :d)) (* :b :f)))
          g (G (fork [:a (slices :x)
                      :i (slices :x)]
                     [:sb1 sb1
                      :sb2 sb2]
                     (* (* :sb1 h) (* :sb2 h) (- (* 2 h) h))
                     [(* :sb1 (- :e :f))
                      (* :sb2 (/ (+ (* :c :d) :e) :b))]))
          inputs {:x [1 2 3] :b 2 :c 1.5 :d 2 :e 3 :f 1}]
      (evaluate g inputs)
      (is (= @init-count 1))))
  (testing "inputs ocurring multiple times in the recurring branches are only initialized once"
    (let [init-count (atom 0)
          h (r/input :h (fn [_]
                          (do (swap! init-count inc)
                              (m/array 1))))
          sb1 (G (+ (* (/ (+ :a :i) 2) (* :b :f)) (* :c :d)))
          sb2 (G (* (+ (/ (+ :a :i) 2) (* :c :d)) (* :b :f)))
          g (G (fork [:a (slices :x)
                      :i (slices :x)]
                     [:sb1 sb1
                      :sb2 sb2]
                     (* :sb1 :sb2)
                     [(* (+ :sb1 h -1) (* (- :e :f) h))
                      (* (* :sb2 h) (/ (+ (* :c :d) :e h -1) :b))]))
          inputs {:x [1 2 3] :b 2 :c 1.5 :d 2 :e 3 :f 1}]
      (evaluate g inputs)
      (is (= @init-count 1))))
  (testing "inputs ocurring in both input and recur initialization branches are initialized only once"
    (let [init-count (atom 0)
          h (r/input :h (fn [_]
                          (do (swap! init-count inc)
                              (m/array 1))))
          g (G (fork [:a (slices (+ (* :x h) h -1))]
                     [:b (* :b (- (* 2 h) 1))]
                     (* :a :b)
                     [(* :b 2)]))
          inputs {:x [3 5 7] :b 4}]
      (evaluate g inputs)
      (is (= @init-count 1))))
  (testing "inputs ocurring in both output and recurring branches are initialized only once"
    (let [init-count (atom 0)
          h (r/input :h (fn [_]
                          (do (swap! init-count inc)
                              (m/array 1))))
          g (G (fork [:a (slices :x)]
                     [:b :b]
                     (* :a (+ :b h -1) h)
                     [(* :b (- 2 h (negate h)) h)]))
          inputs {:x [3 5 7] :b 4}]
      (evaluate g inputs)
      (is (= @init-count 1))))
  (testing "an input present in many branches is initialized only once"
    (let [init-count (atom 0)
          h (r/input :h (fn [_]
                          (do (swap! init-count inc)
                              (m/array 1))))
          sb1 (G (+ (* (/ (+ :a (* :i h)) 2) (* (* :b h) :f)) (* :c :d)))
          sb2 (G (* (+ (/ (+ :a (* :i h)) 2) (* :c :d)) (* :b (* :f h))))
          g (G (fork [:a (slices (+ (* :x h) h -1))
                      :i (slices (* :x (* h h)))]
                     [:sb1 (/ sb1 h)
                      :sb2 (* sb2 h)]
                     (* (* :sb1 h) (- :sb2 (+ h h) -1))
                     [(* :sb1 (- :e (* :f h) h))
                      (* (* :sb2 h) (/ (+ h -1 (* :c :d h) :e) :b))]))
          inputs {:x [1 2 3] :b 2 :c 1.5 :d 2 :e 3 :f 1}]
      (evaluate g inputs)
      (is (= @init-count 1)))))

(deftest nested-fork-test
  (testing "works when another fork is in both the input and the recur initializer branch"
    (let [g1 (G (fork [:a (slices :x)]
                      [:b :b]
                      (* :a :b)
                      [(* :b 3)]))
          g2 (G (fork [:g1 (slices g1)]
                      [:c g1]
                      (* g1 :c)
                      [(* :c 2)]))
          inputs {:x [1 2 3] :b 2}]
      ; g1 evaluates to [2 12 54]
      (evaluates-to? g2 inputs [[4 144 2916]
                                [8 288 5832]
                                [16 576 11664]])
      (numerically-validated? g2 inputs 0.1)))
  (testing "works when another fork is an input and is nested in the output branch"
    (let [g1 (G (fork [:a (slices :x)]
                      [:b :b]
                      (* :a :b)
                      [(* :b 3)]))
          g2 (G (fork [:g1 (slices g1)]
                      [:c :c]
                      (* :g1 :c)
                      [(* :c 2)]))
          inputs {:x [1 2 3] :b 2 :c 3}]
      (evaluates-to? g2 inputs [6 72 648])
      (gradients-are? g2 inputs {:x [6 36 216]
                                 :b 363
                                 :c 242})))
  (testing "works when another fork is in both the input and the recurrent branch, and does not depend on recurrant values"
    (let [g1 (G (fork [:a (slices :x)]
                      [:b :b]
                      (* :a :b)
                      [(* :b 3)]))
          g2 (G (fork [:g1 (slices g1)]
                      [:c :c]
                      (* :c [2 2 2])
                      [(* :c g1)]))
          inputs {:x [1 2 3] :b 2 :c 3}]
      (evaluates-to? g2 inputs [[6 6 6]
                                [12 72 324]
                                [24 864 17496]])
      (numerically-validated? g2 inputs 0.1)))
  (testing "works when another fork is both in the recur initializer and output branch"
    (let [g1 (G (fork [:a (slices :x)]
                      [:b :b]
                      (* :a :b)
                      [(* :b 3)]))
          g2 (G (fork [:c (slices :y)]
                      [:g1 g1]
                      (* :c :g1)
                      [(* :g1 2)]))
          inputs {:x [1 2 3] :b 2 :y [[0 2 1]
                                      [1 0 2]
                                      [2 1 0]]}]
      (evaluates-to? g2 inputs [[0 24 54]
                                [4 0 216]
                                [16 48 0]])
      (gradients-are? g2 inputs {:x [20 36 90]
                                 :b 181
                                 :y [[2 12 54]
                                     [4 24 108]
                                     [8 48 216]]})))
  (testing "works when another fork is both in the recur branch and output branch"
    (let [g1 (G (fork [:a (slices :x)]
                      [:b :b]
                      (* :a :b)
                      [(* :b 3)]))
          g2 (G (fork [:c (slices :y)]
                      [:d :d]
                      (* :c :d g1)
                      [(* g1 2)]))
          inputs {:x [1 2 3] :b 2 :y [[0 2 1]
                                      [1 0 2]
                                      [2 1 0]]
                  :d [2 1 3]}]

      ;; ; Iter 1
      ;; :c [0 2 1]
      ;; :d [2 1 3]
      ;; :g1 {:a [1 2 3]
      ;;      :b [2 6 18]
      ;;      :out [2 12 54]}
      ;; :out [0 24 162]

      ;; ; Iter 2
      ;; :c [1 0 2]
      ;; :d [4 24 108]
      ;; :g1 [2 12 54]
      ;; :out [8 0 11664]

      ;; ; Iter 3
      ;; :c [2 1 0]
      ;; :d [4 24 108]
      ;; :g1 [2 12 54]
      ;; :out [16 288 0]

      ;; [4 0 216]
      ;; [0 2 3]

      ;; :g1-grads
      ;; [[8.0000 24.0000 0]]
      ;; [[12.0000 24.0000 216.0000]]
      ;; [[4.0000 2.0000 219.0000]]

      (evaluates-to? g2 inputs [[0 24 162]
                                [8 0 11664]
                                [16 288 0]])
      (numerically-validated? g2 inputs 0.1)))
  (testing "works when another fork is an input and is used in the recur branch"
    (let [g1 (G (fork [:a (slices :x)]
                      [:d :b]
                      (* :a :d)
                      [(* :d 3)]))
          g2 (G (fork [:g1 (slices g1)]
                      [:c :c]
                      (* :b :c)
                      [(* :g1 2)]))
          inputs {:x [1 2 3] :b 4 :c 3}]
      ; g1 evaluates to [4 24 108]
      (evaluates-to? g2 inputs [12 32 192])
      (numerically-validated? g2 inputs 0.1)))
  (testing "works when a fork is in the input initializer and depends on the input value"
    (let [g1 (G (fork [:d (slices (* :a :x))]
                      [:b :b]
                      (* :d :b)
                      [(* :b 3)]))
          ;; ; Iter 1
          ;; :d [4 2 6]
          ;; :b [2 6 18]
          ;; :g1 [8 12 108]
          ;; :c [8 12 108]

          ;; ; Iter 2
          ;; :d [2 1 3]
          ;; :b [54 162 486]
          ;; :g1 [108 162 1458]
          ;; :c [4 6 54]

          ;; ; Iter 3
          ;; :d [6 3 9]
          ;; :b [1458 4374 13122]
          ;; :g1 [8748 13122 118098]
          ;; :c [2 3 27]

          g2 (G (fork [:a (slices :x)]
                      [:c g1]
                      (* g1 :c)
                      [(/ :c 2)]))
          inputs {:x [2 1 3] :b 2}]
      ; g1 evaluates to [2 12 54]
      (evaluates-to? g2 inputs [[64 144 11664]
                                [432 972 78732]
                                [17496 39366 3188646]])
      (numerically-validated? g2 inputs 0.1)))

  ; TODO: Put the following in "testing" statements
  (let [g1 (G (named :g1 (fork [:d (slices (* :a :x))]
                               [:b :b]
                               (* :d :b)
                               [:b])))
        g2 (G (named :g2 (fork [:a (slices :x)]
                               [:c [8 12 108]]
                               (* g1 :c)
                               [(/ :c 2)])))
        inputs {:x [2 1 3] :b 2}]

    ;; ; Iter 1
    ;; :d [4 2 6]
    ;; :b [2 2 2]
    ;; :g1 [8 4 12]
    ;; :c [8 12 108]
    ;; :g2 [64 48 1224]

    ;; ; Iter 2
    ;; :d [2 1 3]
    ;; :b [2 2 2]
    ;; :g1 [4 2 6]
    ;; :c [4 6 54]
    ;; :g2

    ;; ; Iter 3
    ;; :d [6 3 9]
    ;; :b [2 2 2]
    ;; :g1 [12 6 18]
    ;; :c [2 3 27]
    ;; :g2

    (evaluates-to? g2 inputs [[64 48 1296]
                              [16 12 324]
                              [24 18 486]])
    (numerically-validated? g2 inputs 0.1))

  ; Add another example like belowbut where g1 does not depend on a value from g2
  (let [g1 (G (named :g1 (fork [:d (slices (* :a :x))]
                               [:b :b]
                               (* :d :b)
                               [(* :b 3)])))
        g2 (G (named :g2 (fork [:a (slices :x)]
                               [:c [8 12 108]]
                               (* g1 :c)
                               [(/ :c 2)])))
        inputs {:x [2 1 3] :b 2}]
    ;; ; Iter 1
    ;; :d [4 2 6]
    ;; :b [2 6 18]
    ;; :g1 [8 12 108]
    ;; :c [8 12 108]

    ;; ; Iter 2
    ;; :d [2 1 3]
    ;; :b [54 162 486]
    ;; :g1 [108 162 1458]
    ;; :c [4 6 54]

    ;; ; Iter 3
    ;; :d [6 3 9]
    ;; :b [1458 4374 13122]
    ;; :g1 [8748 13122 118098]
    ;; :c [2 3 27]

    ; g1 evaluates to [2 12 54]
    (evaluates-to? g2 inputs [[64 144 11664]
                              [432 972 78732]
                              [17496 39366 3188646]])
    (numerically-validated? g2 inputs 0.1))

  (let [g1 (G (named :g1 (fork [:d (slices (* :a :x))]
                               [:b :b]
                               :b
                               [(* :b 3)])))
        g2 (G (named :g2 (fork [:a (slices :x)]
                               [:c g1]
                               g1
                               [(/ :c 2)])))
        inputs {:x [2 1 3] :b 2}]
    (numerically-validated? g2 inputs 0.1))
  (let [g1 (G (named :g1 (fork [:d (slices (* :a :x))]
                               [:b (* :b 3)]
                               (* :d :b)
                               [(* :b 3)])))
        g2 (G (named :g2 (fork [:a (slices :x)]
                               [:c [8 12 108]]
                               (* g1 :c)
                               [(/ :c 2)])))
        inputs {:x [2 1 3] :b 2}]
    ;;   ; Iteration 1
    ;; :d [4 2 6]
    ;; :b [6 18 54]
    ;; :g1 [24 36 324]
    ;; :c [8 12 108]
    ;; :g2 [192 432 34992]

    ;; ; Iteration 2
    ;; :b [486 1458 4374]
    ;; :d [2 1 3]
    ;; :g1 [972 1458 13122]
    ;; :c [4 6 54]
    ;; :g2 [3888 8748 708588]

    ;; ; Iteration 3
    ;; :b [39366 118098 354294]
    ;; :d [6 3 9]
    ;; :g1 [236196 354294 3188646]
    ;; :c [2 3 27]
    ;; :g2 [472392 1062882 86093442]

    (evaluates-to? g2 inputs [[192 432 34992]
                              [3888 8748 708588]
                              [472392 1062882 86093442]])
    (numerically-validated? g2 inputs 0.1))
  (let [inner (G (named :g1 (fork [:d (slices (* :a :x))]
                                  [:b (* :b 3)]
                                  (* :d :b)
                                  [(* :b 3)])))
        g2 (G (named :g2 (fork [:a (G (slices (ones [3])))
                                :e (G (slices inner))]
                               [:c inner]
                               (* inner :c)
                               [(/ :c :c2)])))
        ; gradients of :b from inner in body
        ; From first iteration of g2: 13356
        ; From second and third iterations of g2: 2.7714e5
        ; grad of :c [20181 30272 272443]
        ; grad of :b from initializtion of :c: 22461417
        ; (from propagation through body of :g1 to initialize :c 7487139)
        ; backprop from second and third iterations of g2 to

        ;; ; Iteration 1
        ;; :a [1 1 1]
        ;; :e [:d [2 1 3]
        ;;     :b [6 18 54]
        ;;     (* :d :b) [12 18 162]]
        ;; :c [12 18 162]
        ;; :out [144 324 26244]

        ;; ; Iteration 2
        ;; :inner [:d [2 1 3]
        ;;         :b [486 1458 4374]
        ;;         (* :d :b) [972 1458 13122]]
        ;; :c [6 9 81]
        ;; :out [5832 13122 1062882]

        ; Iteration 3
        ;; :inner [:d [2 1 3]
        ;;         :b [39366 118098 354294]
        ;;         (* :d :b) [78732 118098 1062882]]
        ;; :c [3 4.5 40.5]
        ;; :out [236196 531441 43046721]

        inputs {:x [2 1 3]
                :b 2
                :c2 2}]
    (evaluates-to? g2 inputs [[144 324 26244]
                              [5832 13122 1062882]
                              [236196 531441 43046721]])
    (numerically-validated? g2 inputs 0.1))
  (let [inner (G (named :g1 (fork [:d (slices (* :a [1 1 1]))]
                                  [:b (* :b 3)]
                                  (* :d :b)
                                  [(* :b 3)])))
        g2 (G (named :g2 (fork [:a (G (slices :y))]
                               [:c [1 2 3]]
                               (pow :c 2)
                               [inner])))
        inputs {:y [40 20 10]
                :b 2}
        vals (forward g2 inputs)
        grads (backward g2 vals)]

    ;; ; Iteration 1
    ;; :a 40
    ;; :c [1 2 3]
    ;; :output [1 4 9]
    ;; :inner [:d [40 40 40]
    ;;         :b [6 18 54]
    ;;         (* :b :d) [240 720 2160]
    ;;         ]

    ;; ; Iteration 2
    ;; :a 20
    ;; :c [240 720 2160]
    ;; :output [57600 518400 4665600]
    ;; :inner [:d [20 20 20]
    ;;         :b [486 1458 4374]
    ;;         (* :b :d) [9720 29160 87480]]

    ;; ; Iteration 3
    ;; :a 10
    ;; :c [9720 29160 87480]
    ;; :output [94478400 850305600 7652750400]

    (evaluates-to? g2 inputs [[1 4 9]
                              [57600 518400 4665600]
                              [94478400 850305600 7652750400]])
    ; TODO: Get this test passing
    #_(numerically-validated? g2 inputs 0.1)))

; TODO: Add more combinations to this test.
(deftest nested-fork-combinations-test
  (let [inners [(G (named :g1 (fork [:d (slices (* :a :x))]
                                    [:b (* :b 3)]
                                    (* :d :b)
                                    [(* :b 3)])))]
        input-initializers (concat [[:a (G (slices (* :x :c3)))
                                     :e (G (slices (ones [3])))]
                                    [:a (map m/array [2 1 3])
                                     :e (G (slices (ones [3])))]
                                    [:a (G (slices (* :x :c3)))
                                     :e (G (slices (* :x :c3)))]
                                    [:a (G (slices (* :x :c3)))
                                     :e (G (slices (* :y :c3)))]
                                    [:a (G (slices (* :x :c3)))
                                     :e (G (slices (* :y :c4)))]]
                                   (mapcat (fn [inner]
                                             [[:a (G (slices (ones [3])))
                                               :e (G (slices inner))]
                                              ; TODO: Get these tests passing (these failures are related to
                                              ; the failing backward test result for the last test in
                                              ; `nested-fork-test`.
                                              #_[:a (G (slices :y))
                                                 :e (G (slices inner))]])
                                           inners))
        recur-initializers (concat [[:c [1 2 3]]]
                                   (mapcat (fn [inner]
                                             [[:c (G (* inner :c2))]
                                              [:c inner]])
                                           inners))
        outputs (mapcat (fn [inner]
                          [(G :c)
                           (G (pow :c 2))
                           (G inner)
                           (G (* inner :c))
                           (G :e)
                           (G :a)
                           (G (* :e :a))
                           (G (* (* :e :a) :c1))
                           (G (* (* :e :a) (+ :a (* :c1 :e))))])
                        inners)
        recur-branches (concat [[(G (/ :c :c2))]]
                               (mapcat (fn [inner]
                                         [[(G (/ inner :c2))]])
                                       inners))
        inputs {:x [2 1 3]
                :y [4 2 1]
                :z [[1 2 3]
                    [4 5 6]
                    [7 8 9]]
                :b 2
                :c1 3
                :c2 2
                :c3 1
                :c4 1}]
    (doseq [input-initializer input-initializers
            recur-initializer recur-initializers
            output outputs
            recur-branch recur-branches]
      (numerically-validated?
        (G (named :g2 (fork input-initializer
                            recur-initializer
                            output
                            recur-branch)))
        inputs
        0.1))))

#_(clojure.test/run-tests)
