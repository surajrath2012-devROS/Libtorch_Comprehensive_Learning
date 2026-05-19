#include "Sensor.h"
#include "ThreadQueue.h"
#include "NeuralBrain.h"
#include <iostream>
#include <memory>
#include <thread>
#include <chrono>

// Our shared queue that holds LibTorch Tensors
ThreadQueue<torch::Tensor> frame_queue;

void camera_producer_thread() {
    std::cout << "[Producer] Thread started." << std::endl;
    std::unique_ptr<BaseSensor> camera = std::make_unique<MnistCamera>("../data/mnist/train-images-idx3-ubyte");

    // Simulate capturing 5 frames at 30 FPS
    for (int i = 0; i < 5; i++) {
        torch::Tensor frame = camera->capture_frame();
        frame_queue.push(frame);
        std::cout << "[Producer] Captured and pushed frame " << i << std::endl;
        
        // Sleep for 33ms to simulate 30 FPS hardware
        std::this_thread::sleep_for(std::chrono::milliseconds(33)); 
    }

    std::cout << "[Producer] Hardware feed finished. Shutting down queue." << std::endl;
    frame_queue.shutdown(); // Tell the consumer to go home
}

void ai_consumer_thread() {
    std::cout << "[Consumer] Thread started. Waking up Neural Brain..." << std::endl;
    
    // 1. Instantiate our C++ AI Engine
    NeuralBrain brain;
    brain.eval(); // Put the network into inference/evaluation mode

    std::cout << "[Consumer] Brain ready. Waiting for sensor data..." << std::endl;

    while (true) {
        std::optional<torch::Tensor> opt_frame = frame_queue.pop();
        
        if (!opt_frame.has_value()) {
            std::cout << "[Consumer] Queue shut down. Exiting thread." << std::endl;
            break;
        }

        torch::Tensor frame = opt_frame.value();
        
        // 2. The AI consumes the frame and makes a prediction
        int predicted_digit = brain.predict_digit(frame);
        
        std::cout << "[Consumer] AI Prediction: " << predicted_digit << std::endl;
        
        // Simulate hardware processing time
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

int main() {
    std::cout << "--- Initializing Asynchronous Vision Pipeline ---\n" << std::endl;

    // 1. Spawn the threads (they start running immediately at the same time)
    std::thread producer(camera_producer_thread);
    std::thread consumer(ai_consumer_thread);

    // 2. The main program must wait for both threads to finish their work before exiting
    producer.join();
    consumer.join();

    std::cout << "\n--- Pipeline Shutdown Safely ---" << std::endl;
    return 0;
}