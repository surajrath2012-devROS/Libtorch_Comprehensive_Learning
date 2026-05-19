#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>

/**
 * @class ThreadQueue
 * @brief A thread-safe, generic queue for Producer-Consumer pipelines.
 */
template <typename T>
class ThreadQueue {
private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cond_var_;
    bool shutdown_flag_ = false;

public:
    /**
     * @brief Pushes an item into the queue and wakes up a sleeping consumer.
     */
    void push(T item) {
        // The {} scope lock ensures the mutex unlocks the microsecond this function ends
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(item)); // std::move transfers ownership without copying heavy data
        }
        cond_var_.notify_one(); // "Hey Consumer, wake up! New data is here."
    }

    /**
     * @brief Pulls an item. If empty, the thread goes to sleep until data arrives.
     * @return std::optional<T> returns the item, or nullopt if shutting down.
     */
    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        
        // Go to sleep UNTIL the queue is not empty, OR we are told to shut down
        cond_var_.wait(lock, [this]() { return !queue_.empty() || shutdown_flag_; });

        if (shutdown_flag_ && queue_.empty()) {
            return std::nullopt; // System is turning off, return nothing safely
        }

        T item = std::move(queue_.front());
        queue_.pop();
        return item;
    }

    /**
     * @brief Signals all sleeping threads to wake up and exit safely.
     */
    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            shutdown_flag_ = true;
        }
        cond_var_.notify_all();
    }
};