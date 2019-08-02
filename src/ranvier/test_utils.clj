(ns ranvier.test-utils
  (:require [clojure.test :refer [is testing]]
            [ranvier.core :refer [input get-node-value forward backward evaluate G get-node-name
                                  get-gradients get-all-graph-input-names G]]
            [ranvier.utils :as u]
            [tensure.core :as m]))

; Taken from this thread: https://groups.google.com/forum/#!topic/clojure/mJqplAdt3TY
; Inspired by midje's testable-privates fn.
(defn refer-privates
  "Interns the given private symbols into the current namespace. Usage:
   (refer-privates 'some-other-ns :private-fn1 :private-fn2)"
  [ns & symbols-as-keywords]
  (let [symbols (->> symbols-as-keywords (map #(symbol (name %))))
        private-symbols (into {} (filter #(-> % val meta :private) (ns-interns ns)))]
    (when-let [missing-symbol (first (filter #(not (contains? private-symbols %)) symbols))]
      (u/throw-str "There is no private symbol '" missing-symbol "' in ns '" ns "'."))
    (doseq [symbol symbols]
      (when-let [var (get private-symbols symbol)]
        (intern *ns* symbol var)))))

(defn nd=
  "Converst `a` and `b` to tensors and returns true iff they are equal."
  [a b]
  (m/equals (m/array a) (m/array b)))

(defn about=
  "Returns true iff `a` and `b` are scalars/matrices of the same shape where every element is a number and
  `a - b` < `tolerance`."
  ([a b]
   (about= a b 0.00001))
  ([a b tolerance]
   (let [a (m/array a)
         b (m/array b)]
     (if (not= (m/shape a) (m/shape b))
       false
       (->> (m/sub a b)
            m/abs
            (#(m/le % (m/array tolerance)))
            m/eseq
            (every? #(nd= 1 %)))))))

(defn map-values-about=
  "Given maps `a` and `b`, returns true iff they have the same keys and all values for the equivalent key
  differ by no more than `tolerance` (fractional difference of the mean of the `a` and `b` values)."
  ([a b]
   (map-values-about= a b nil))
  ([a b tolerance]
   (every? (fn [k]
             (let [a-val (->> (get a k)
                              (#(if (or (m/array? %) (nil? %))
                                  %
                                  (m/array %))))
                   b-val (->> (get b k)
                              (#(if (or (m/array? %) (nil? %))
                                  %
                                  (m/array %))))]
               (or (= a-val b-val)
                   (and a-val b-val
                        (if tolerance
                          (let [allowed-diff (-> (m/add a-val b-val)
                                                 (m/div (m/array 2))
                                                 (m/mul (m/array tolerance))
                                                 m/abs)]
                            (about= a-val b-val allowed-diff))
                          (about= a-val b-val))))))
           (keys a))))

(defn test-op
  "Given an operation defined with `defop`, the gradient of the result (`dv`), and any number of arguments to
  the op (`operands`), returns a vec like `[val operand-gradients]`, where `val` is the result of evaluating
  `(apply op operands)` and `operand-gradients` is a vector of the backpropagated gradients of all operands."
  [op dv & operands]
  (let [input-names (repeatedly (count operands) #(keyword (gensym "operand")))
        input-map (->> (map #(vector %1 %2) input-names operands)
                       (into {}))
        g (apply op (map input input-names))
        values (forward g input-map)
        root-node-name (get-node-name g)
        root-grad (if (m/array? (get values root-node-name))
                    (m/array dv)
                    dv)
        gradients (backward g values input-names root-grad {:return-all false})
        operand-gradients (map #(get gradients %) input-names)
        val (get values root-node-name)]
    [val operand-gradients]))

(defn test-op-with-args
  "Like `test-op`, except only the first operand `a` to `op` is treated as an input. The remaining `args` are
  passed directly to `op` as standard function arguments."
  [op dv a & args]
  (let [g (apply op (input :a) args)
        values (forward g {:a (m/array a)})
        val (get values (get-node-name g))
        {da :a} (backward g values [:a] (m/array dv))]
    [val da]))

(defn numeric-grad
  "Given a graph `g`, an `input-map` (name->value), and a seq of `input-names`, returns a map of `input-grads`,
  which contains gradients of the final graph output with respect to each input in `input-names`. If the final
  output of `g` is non-scalar, the gradients can be interpreted as those for the sum of all elements (across
  dimensions) of the output."
  [g input-map input-names]
  (let [dx 1e-3
        y1 (m/esum (evaluate g input-map))]
    (->> (mapv (fn [input-name]
                 (let [input-val (-> (get input-map input-name)
                                     (#(if (m/array? %)
                                         %
                                         (m/array %))))]
                   [input-name
                    (m/emap-indexed (fn [index-vec el-val]
                                      (let [x2 (apply m/mset input-val (concat index-vec [(+ el-val dx)]))
                                            y2 (m/esum (evaluate g (assoc input-map input-name x2)))
                                            dy (m/scalar->number (m/sub y2 y1))]
                                        (/ dy dx)))
                                    input-val)]))
               input-names)
         (into {}))))

(defn numerically-validated?
  "Given a graph `g` and a map of `inputs`, asserts that automatic differentiation (backprop) returns the same
  gradients as those determined numerically using `numeric-grads`. Griadients that differ fractionally by less
  than `tolerance` are considered the same. This can be used to determine the correctness of backprop, but it
  assumes forward prop gives the correct result."
  ([g]
   (numerically-validated? g (forward g {})))
  ([g inputs]
   (numerically-validated? g inputs 0.01))
  ([g inputs tolerance]
   (let [autodiff-result (->> (forward g inputs)
                              (backward g)
                              (#(select-keys % (keys inputs))))
         numeric-result (numeric-grad g inputs (keys autodiff-result))]
     (is (map-values-about= autodiff-result numeric-result tolerance)))))

(defn params-numerically-validated?
  "Like `numerically-validated?` but checks gradients with respect to all parameters of `g` in addition to
  all `inputs`."
  ([g inputs]
   (params-numerically-validated? g inputs 0.01))
  ([g inputs tolerance]
   (let [node-names-to-validate (concat (get-all-graph-input-names g) (keys inputs))
         values (-> (forward g inputs)
                    (select-keys node-names-to-validate))]
     (numerically-validated? g values tolerance))))

; TODO: Make this more user friendly, perhaps by making it return a boolean and pulling the `is` outside
; the function call wherever this is used, or else by turning this into a macro; with the current approach,
; failures reulst in no useful information about which test is failing.
(defn evaluates-to?
  "Given a graph `g`, an optional map of `inputs`, and an expected `result`, asserts that running `g` on
  `inputs` will return `result`."
  ([g expected-result]
   (evaluates-to? g {} expected-result))
  ([g inputs expected-result]
   (let [actual-result (evaluate g inputs)
         [comparison-op expected-result] (if (m/array? actual-result)
                                           [m/equals (m/array expected-result)]
                                           [= expected-result])]
     (is (comparison-op actual-result expected-result)))))

(defn evaluates-to-about?
  "Like `evaluates-to?` but performs an approximate comparison using `about=` with the given `tolerance`."
  ([g result]
   (evaluates-to? g {} result))
  ([g inputs result]
   (evaluates-to-about? g inputs result 0.01))
  ([g inputs result tolerance]
   (is (about= (evaluate g inputs) result tolerance))))

(defn gradients-are?
  "Asserts that gradients of `g` at `inputs` are about= to their values in `result` (a map of input name
  to gradient value), to tolerance `tolerance`. Running backpropagation on `g` can produce gradients for nodes
  not in `result`, and these are not checked."
  ([g result]
   (gradients-are? g {} result))
  ([g inputs result]
   (gradients-are? g inputs result 0.001))
  ([g inputs result tolerance]
   (is (map-values-about= result (select-keys (get-gradients g inputs) (keys result)) tolerance))))

(defn check-tensor-creator
  "Asserts that the provided `tensor-creator` function returns a tensor of the same shape as its first
  argument."
  [tensor-creator]
  (testing "returns a tensor of the correct shape"
    #_(is (= (m/shape (tensor-creator nil)) nil))
    (is (= (m/shape (tensor-creator [3])) [3]))
    (is (= (m/shape (tensor-creator [2 3])) [2 3]))
    (is (= (m/shape (tensor-creator [2 3 4])) [2 3 4]))))

(defn uniform-sd
  "Returns the standard deviation of a uniform distribution over [min, max)."
  [min max]
  (Math/sqrt (/ (Math/pow (- max min) 2) 12)))

(defn truncated-normal-sd
  [mean stdev]
  (let [limit (* 2 stdev)
        min (- mean limit)
        max (+ mean limit)
        alpha 0.5
        beta 0.5]))

(defn check-op-backpropagates-nil
  "Given a two-arity op that accepts matrices, asserts that gradients backpropagated to both inputs are `nil`."
  [op]
  (testing "backpropagates nil"
    (let [[_ [da db]] (test-op op [[1 2] [3 4]] [[1 0] [0 1]] [[1 1] [0 0]])]
      (is (nil? da))
      (is (nil? db)))))

(defn between?
  "Returns `true` iff `n` is in [lower, upper)."
  [lower n upper]
  (and (>= n lower)
       (< n upper)))

(defn random-shape
  "Returns a vector of integers representing a random tensor shape, given:
    - `min-dimesionality` (optional, default 0 (i.e. the default includes scalars))
    - `max-dimensionality` (optional, default 5)
    - `max-dimension-size` (optional, default 5)"
  ([]
   (random-shape 5 5))
  ([max-dimensionality max-dimension-size]
   (random-shape 0 max-dimensionality max-dimension-size))
  ([min-dimensionality max-dimensionality max-dimension-size]
   (let [dimensionality (u/rand-int-between min-dimensionality (inc max-dimensionality))
         gen-dimension #(u/rand-int-between 1 (inc max-dimension-size))]
     (repeatedly dimensionality gen-dimension))))
