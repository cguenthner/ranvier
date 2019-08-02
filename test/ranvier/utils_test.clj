(ns ranvier.utils-test
  (:require [clojure.test :refer :all]
            [ranvier.utils :refer :all]))

(deftest zip-test
  (is (= (zip [0])) [0])
  (is (= (zip [0] [1]) [[0 1]]))
  (is (= (zip [0] [1] [2]) [[0 1 2]]))
  (is (= (zip [0 1] [2 3]) [[0 2] [1 3]]))
  (is (= (zip [0 1] [2 3] [4 5]) [[0 2 4] [1 3 5]]))
  (is (= (zip [0 1] [2]) [[0 2]]))
  (is (= (zip [0] [1 2 3])) [[0 1]])
  (is (= (zip [] [1])) [])
  (is (= (zip [] [])) [])
  (is (= (zip) [])))

(deftest check-key-validity-test
  (let [m {:a 1 :b 2 :c 3 :d 4}]
    (testing "Accepts valid maps"
      (is (nil? (check-key-validity m [:a :b :c :d] [])))
      (is (nil? (check-key-validity m [:a :b :c] [:d])))
      (is (nil? (check-key-validity m [:a :b :c :d] [:e])))
      (is (nil? (check-key-validity m [] [:a :b :c :d])))
      (is (nil? (check-key-validity {} [] [:a :b :c :d]))))
    (testing "Doesn't allow missing keys"
      (is (thrown? Exception (check-key-validity m [:a :b :c] []))))
    (testing "Doesn't allow invalid keys"
      (is (thrown? Exception (check-key-validity (merge m {:e 5}) [:a :b :c :d] [])))
      (is (thrown? Exception (check-key-validity (merge m {:e 5}) [:a :b :c] [:d]))))))

; NOTE: This is non-determinist (it may fail occassionally just by chance, but it is very unlikely)
(deftest rand-int-between-test
  (let [sample-count 1e4
        get-sample-range (fn [lower upper]
                           (->> (repeatedly sample-count #(rand-int-between lower upper))
                                distinct
                                ((juxt #(apply min %) #(apply max %)))))]
    (let [[min-val max-val] (get-sample-range 0 5)]
      (is (= min-val 0))
      (is (= max-val 4)))
    (let [[min-val max-val] (get-sample-range -5 0)]
      (is (= min-val -5))
      (is (= max-val -1)))
    (let [[min-val max-val] (get-sample-range -3 2)]
      (is (= min-val -3))
      (is (= max-val 1)))))
