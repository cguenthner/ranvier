(ns ranvier.test-utils-test
  (:require [clojure.test :refer :all]
            [ranvier.test-utils :refer :all]
            [ranvier.utils :as u]
            [tensure.core :as m]))

; NOTE: This could theoretically fail on occasion by chance, but this is very unlikely.
(defn check-random-shape
  ([]
   (check-random-shape 5 5))
  ([max-dimensionality max-dimension-size]
   (check-random-shape 0 max-dimensionality max-dimension-size))
  ([min-dimensionality max-dimensionality max-dimension-size]
   (let [results (repeatedly 1e4 #(random-shape min-dimensionality
                                                max-dimensionality max-dimension-size))
         dimensionalities (mapv count results)
         all-dimension-sizes (apply concat results)]
     (is (= (apply min dimensionalities) min-dimensionality))
     (is (= (apply max dimensionalities) max-dimensionality))
     (if (zero? max-dimensionality)
       (is (= all-dimension-sizes []))
       (do (is (= (apply min all-dimension-sizes) 1))
           (is (= (apply max all-dimension-sizes) max-dimension-size)))))))

(deftest random-shape-test
  (check-random-shape)
  (check-random-shape 0 0)
  (check-random-shape 0 5)
  (check-random-shape 1 1)
  (check-random-shape 1 5)
  (check-random-shape 5 1)
  (check-random-shape 5 5)
  (check-random-shape 0 0 1)
  (check-random-shape 0 0 5)
  (check-random-shape 0 5 1)
  (check-random-shape 0 5 5)
  (check-random-shape 1 1 5)
  (check-random-shape 1 5 5))
