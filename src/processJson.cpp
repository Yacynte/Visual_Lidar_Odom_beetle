#include "processJson.h"

void processJsonString(const std::string& jsonStr) {
    try {
        json j = json::parse(jsonStr);
        std::cout << "Received image timestamps:\n";

        for (const auto& [timestamp, value] : j.items()) {
            bool detected = value.get<bool>();
            std::cout << "  " << timestamp << " => " << (detected ? "Detected" : "Not Detected") << "\n";
        }

        // Optional: prepare response
        int true_count = std::count_if(j.begin(), j.end(),
            [](const auto& item) { return item.second.get<bool>(); });

        json response = {
            {"status", "ok"},
            {"detected_count", true_count}
        };

        std::cout << "Responding with: " << response.dump() << "\n";

        // send(response.dump().c_str(), ...) ← in actual socket code
    }
    catch (json::parse_error& e) {
        std::cerr << "JSON Parse Error: " << e.what() << "\n";
    }
}
