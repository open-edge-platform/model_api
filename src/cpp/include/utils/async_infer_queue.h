#include <condition_variable>
#include <functional>
#include <openvino/openvino.hpp>
#include <queue>
#include <vector>

class AsyncInferQueue {
public:
    AsyncInferQueue() = default;
    AsyncInferQueue(ov::CompiledModel& model, size_t jobs);
    ~AsyncInferQueue() = default;

    bool is_ready();
    size_t get_idle_request_id();
    void wait_all();
    void set_default_callbacks();
    void set_custom_callbacks(
        std::function<void(ov::InferRequest, std::shared_ptr<ov::AnyMap> callback_args)> f_callback);
    size_t size() const;
    void start_async(const ov::Tensor& input, std::shared_ptr<ov::AnyMap> userdata = nullptr);
    void start_async(const std::map<std::string, ov::Tensor>& input, std::shared_ptr<ov::AnyMap> userdata = nullptr);
    ov::InferRequest operator[](size_t i);

    // AsyncInferQueue is not the sole owner of infer requests, although it calls create_infer_request() method.
    // ov::InferRequest contains a shared pointer internally, and therefore requests can be copied by clients of
    // AsyncInferQueue.
protected:
    std::vector<ov::InferRequest> m_requests;
    std::queue<size_t> m_idle_handles;
    std::vector<std::shared_ptr<ov::AnyMap>> m_user_ids;
    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::queue<std::shared_ptr<std::exception>> m_errors;
};
