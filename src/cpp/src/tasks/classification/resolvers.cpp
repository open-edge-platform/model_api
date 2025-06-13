#include <nlohmann/json.hpp>
#include <deque>

#include "tasks/classification/resolvers.h"

HierarchicalConfig::HierarchicalConfig(const std::string& json_repr) {
    nlohmann::json data = nlohmann::json::parse(json_repr);

    num_multilabel_heads = data.at("cls_heads_info").at("num_multilabel_classes");
    num_multiclass_heads = data.at("cls_heads_info").at("num_multiclass_heads");
    num_single_label_classes = data.at("cls_heads_info").at("num_single_label_classes");

    data.at("cls_heads_info").at("label_to_idx").get_to(label_to_idx);
    data.at("cls_heads_info").at("all_groups").get_to(all_groups);
    data.at("label_tree_edges").get_to(label_tree_edges);

    std::map<std::string, std::pair<int, int>> tmp_head_idx_to_logits_range;
    data.at("cls_heads_info").at("head_idx_to_logits_range").get_to(tmp_head_idx_to_logits_range);

    for (const auto& range_descr : tmp_head_idx_to_logits_range) {
        head_idx_to_logits_range[stoi(range_descr.first)] = range_descr.second;
    }

    size_t logits_processed = 0;
    for (size_t i = 0; i < num_multiclass_heads; ++i) {
        const auto& logits_range = head_idx_to_logits_range[i];
        for (size_t k = logits_range.first; k < logits_range.second; ++k) {
            logit_idx_to_label[logits_processed++] = all_groups[i][k - logits_range.first];
        }
    }
    for (size_t i = 0; i < num_multilabel_heads; ++i) {
        logit_idx_to_label[logits_processed++] = all_groups[num_multiclass_heads + i][0];
    }
}

GreedyLabelsResolver::GreedyLabelsResolver(const HierarchicalConfig& config)
    : label_to_idx(config.label_to_idx),
      label_relations(config.label_tree_edges),
      label_groups(config.all_groups) {}

std::map<std::string, float> GreedyLabelsResolver::resolve_labels(
    const std::vector<std::reference_wrapper<std::string>>& labels,
    const std::vector<float>& scores) {
    if (labels.size() != scores.size()) {
        throw std::runtime_error("Inconsistent number of labels and scores");
    }
    std::map<std::string, float> label_to_prob;
    for (const auto& label_idx : label_to_idx) {
        label_to_prob[label_idx.first] = 0.f;
    }

    for (size_t i = 0; i < labels.size(); ++i) {
        label_to_prob[labels[i]] = scores[i];
    }

    std::vector<std::string> candidates;
    for (const auto& g : label_groups) {
        if (g.size() == 1 && label_to_prob[g[0]] > 0.f) {
            candidates.push_back(g[0]);
        } else {
            float max_prob = 0.f;
            std::string max_label;
            for (const auto& lbl : g) {
                if (label_to_prob[lbl] > max_prob) {
                    max_prob = label_to_prob[lbl];
                    max_label = lbl;
                }
                if (max_label.size() > 0) {
                    candidates.push_back(max_label);
                }
            }
        }
    }

    std::map<std::string, float> resolved_label_to_prob;
    for (const auto& lbl : candidates) {
        if (resolved_label_to_prob.find(lbl) != resolved_label_to_prob.end()) {
            continue;
        }
        auto labels_to_add = get_predecessors(lbl, candidates);
        for (const auto& new_lbl : labels_to_add) {
            if (resolved_label_to_prob.find(new_lbl) == resolved_label_to_prob.end()) {
                resolved_label_to_prob[new_lbl] = label_to_prob[new_lbl];
            }
        }
    }

    return resolved_label_to_prob;
}

std::string GreedyLabelsResolver::get_parent(const std::string& label) {
    for (const auto& edge : label_relations) {
        if (label == edge.first) {
            return edge.second;
        }
    }
    return "";
}

std::vector<std::string> GreedyLabelsResolver::get_predecessors(const std::string& label,
                                                                const std::vector<std::string>& candidates) {
    std::vector<std::string> predecessors;
    auto last_parent = get_parent(label);

    if (last_parent.size() == 0) {
        return {label};
    }
    while (last_parent.size() > 0) {
        if (std::find(candidates.begin(), candidates.end(), last_parent) == candidates.end()) {
            return {};
        }
        predecessors.push_back(last_parent);
        last_parent = get_parent(last_parent);
    }

    if (predecessors.size() > 0) {
        predecessors.push_back(label);
    }

    return predecessors;
}

SimpleLabelsGraph::SimpleLabelsGraph(const std::vector<std::string>& vertices_)
    : vertices(vertices_),
      t_sort_cache_valid(false) {}

void SimpleLabelsGraph::add_edge(const std::string& parent, const std::string& child) {
    adj[parent].push_back(child);
    parents_map[child] = parent;
    t_sort_cache_valid = false;
}

std::vector<std::string> SimpleLabelsGraph::get_children(const std::string& label) const {
    auto iter = adj.find(label);
    if (iter == adj.end()) {
        return std::vector<std::string>();
    }
    return iter->second;
}

std::string SimpleLabelsGraph::get_parent(const std::string& label) const {
    auto iter = parents_map.find(label);
    if (iter == parents_map.end()) {
        return std::string();
    }
    return iter->second;
}

std::vector<std::string> SimpleLabelsGraph::get_ancestors(const std::string& label) const {
    std::vector<std::string> predecessors = {label};
    auto last_parent = get_parent(label);
    if (!last_parent.size()) {
        return predecessors;
    }

    while (last_parent.size()) {
        predecessors.push_back(last_parent);
        last_parent = get_parent(last_parent);
    }

    return predecessors;
}

std::vector<std::string> SimpleLabelsGraph::get_labels_in_topological_order() {
    if (!t_sort_cache_valid) {
        topological_order_cache = topological_sort();
    }
    return topological_order_cache;
}

std::vector<std::string> SimpleLabelsGraph::topological_sort() {
    auto in_degree = std::unordered_map<std::string, size_t>();
    for (const auto& node : vertices) {
        in_degree[node] = 0;
    }

    for (const auto& item : adj) {
        for (const auto& node : item.second) {
            in_degree[node] += 1;
        }
    }

    std::deque<std::string> nodes_deque;
    for (const auto& node : vertices) {
        if (in_degree[node] == 0) {
            nodes_deque.push_back(node);
        }
    }

    std::vector<std::string> ordered_nodes;
    while (!nodes_deque.empty()) {
        auto u = nodes_deque[0];
        nodes_deque.pop_front();
        ordered_nodes.push_back(u);

        for (const auto& node : adj[u]) {
            auto degree = --in_degree[node];
            if (degree == 0) {
                nodes_deque.push_back(node);
            }
        }
    }

    if (ordered_nodes.size() != vertices.size()) {
        throw std::runtime_error("Topological sort failed: input graph has been"
                                 "changed during the sorting or contains a cycle");
    }

    return ordered_nodes;
}

ProbabilisticLabelsResolver::ProbabilisticLabelsResolver(const HierarchicalConfig& conf) : GreedyLabelsResolver(conf) {
    std::vector<std::string> all_labels;
    for (const auto& item : label_to_idx) {
        all_labels.push_back(item.first);
    }
    label_tree = SimpleLabelsGraph(all_labels);
    for (const auto& item : label_relations) {
        label_tree.add_edge(item.second, item.first);
    }
    label_tree.get_labels_in_topological_order();
}

std::map<std::string, float> ProbabilisticLabelsResolver::resolve_labels(
    const std::vector<std::reference_wrapper<std::string>>& labels,
    const std::vector<float>& scores) {
    if (labels.size() != scores.size()) {
        throw std::runtime_error("Inconsistent number of labels and scores");
    }

    std::unordered_map<std::string, float> label_to_prob;
    for (size_t i = 0; i < labels.size(); ++i) {
        label_to_prob[labels[i]] = scores[i];
    }

    label_to_prob = add_missing_ancestors(label_to_prob);
    auto hard_classification = resolve_exclusive_labels(label_to_prob);
    suppress_descendant_output(hard_classification);

    std::map<std::string, float> output_labels_map;

    for (const auto& item : hard_classification) {
        if (item.second > 0) {
            if (label_to_prob.find(item.first) != label_to_prob.end()) {
                output_labels_map[item.first] = item.second * label_to_prob[item.first];
            } else {
                output_labels_map[item.first] = item.second;
            }
        }
    }

    return output_labels_map;
}

std::unordered_map<std::string, float> ProbabilisticLabelsResolver::add_missing_ancestors(
    const std::unordered_map<std::string, float>& label_to_prob) const {
    std::unordered_map<std::string, float> updated_label_to_probability(label_to_prob);
    for (const auto& item : label_to_prob) {
        for (const auto& ancestor : label_tree.get_ancestors(item.first)) {
            if (updated_label_to_probability.find(ancestor) == updated_label_to_probability.end()) {
                updated_label_to_probability[ancestor] = 0.f;
            }
        }
    }
    return updated_label_to_probability;
}

std::map<std::string, float> ProbabilisticLabelsResolver::resolve_exclusive_labels(
    const std::unordered_map<std::string, float>& label_to_prob) const {
    std::map<std::string, float> hard_classification;

    for (const auto& item : label_to_prob) {
        hard_classification[item.first] = static_cast<float>(item.second > 0);
    }

    return hard_classification;
}

void ProbabilisticLabelsResolver::suppress_descendant_output(std::map<std::string, float>& hard_classification) {
    auto all_labels = label_tree.get_labels_in_topological_order();

    for (const auto& child : all_labels) {
        if (hard_classification.find(child) != hard_classification.end()) {
            auto parent = label_tree.get_parent(child);
            if (parent.size() && hard_classification.find(parent) != hard_classification.end()) {
                hard_classification[child] *= hard_classification[parent];
            }
        }
    }
}
