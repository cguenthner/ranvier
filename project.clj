(defproject ranvier "0.1.0"
  :description "Clojure numerical optimization and machine learning library"
  :url "https://github.com/cguenthner/ranvier"
  :license {:name "MIT License"
            :url "https://opensource.org/licenses/MIT"}
  :dependencies [[org.clojure/clojure "1.9.0"]
                 [tensure "0.1.0"]
                 [org.clojure/core.async "0.4.490"]]
  :repl-options {:init-ns ranvier.core}
  :codox {:namespaces [ranvier.core]
          :metadata {:doc/format :markdown}
          :source-uri "https://github.com/cguenthner/ranvier/blob/{git-commit}/{filepath}#L{line}"})
