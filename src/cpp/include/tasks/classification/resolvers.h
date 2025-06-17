/*
 * Copyright (C) 2020-2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

struct HierarchicalConfig {
    std::map<std::string, int> label_to_idx;
    std::vector<std::pair<std::string, std::string>> label_tree_edges;
    std::vector<std::vector<std::string>> all_groups;
    std::map<size_t, std::pair<size_t, size_t>> head_idx_to_logits_range;
    std::map<size_t, std::string> logit_idx_to_label;
    size_t num_multiclass_heads;
    size_t num_multilabel_heads;
    size_t num_single_label_classes;

    HierarchicalConfig() = default;
    HierarchicalConfig(const std::string&);
};

class SimpleLabelsGraph {
public:
    SimpleLabelsGraph() = default;
    SimpleLabelsGraph(const std::vector<std::string>& vertices_);
    void add_edge(const std::string& parent, const std::string& child);
    std::vector<std::string> get_children(const std::string& label) const;
    std::string get_parent(const std::string& label) const;
    std::vector<std::string> get_ancestors(const std::string& label) const;
    std::vector<std::string> get_labels_in_topological_order();

protected:
    std::vector<std::string> vertices;
    std::unordered_map<std::string, std::vector<std::string>> adj;
    std::unordered_map<std::string, std::string> parents_map;
    bool t_sort_cache_valid = false;
    std::vector<std::string> topological_order_cache;

    std::vector<std::string> topological_sort();
};

class GreedyLabelsResolver {
public:
    GreedyLabelsResolver() = default;
    GreedyLabelsResolver(const HierarchicalConfig&);

    virtual std::map<std::string, float> resolve_labels(const std::vector<std::reference_wrapper<std::string>>& labels,
                                                        const std::vector<float>& scores);

protected:
    std::map<std::string, int> label_to_idx;
    std::vector<std::pair<std::string, std::string>> label_relations;
    std::vector<std::vector<std::string>> label_groups;

    std::string get_parent(const std::string& label);
    std::vector<std::string> get_predecessors(const std::string& label, const std::vector<std::string>& candidates);
};

class ProbabilisticLabelsResolver : public GreedyLabelsResolver {
public:
    ProbabilisticLabelsResolver() = default;
    ProbabilisticLabelsResolver(const HierarchicalConfig&);

    virtual std::map<std::string, float> resolve_labels(const std::vector<std::reference_wrapper<std::string>>& labels,
                                                        const std::vector<float>& scores);
    std::unordered_map<std::string, float> add_missing_ancestors(const std::unordered_map<std::string, float>&) const;
    std::map<std::string, float> resolve_exclusive_labels(const std::unordered_map<std::string, float>&) const;
    void suppress_descendant_output(std::map<std::string, float>&);

protected:
    SimpleLabelsGraph label_tree;
};
