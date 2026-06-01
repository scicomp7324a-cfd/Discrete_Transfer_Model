#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

class PathConfig {
private:
    fs::path inputBaseDir_;
    fs::path outputBaseDir_;

public:
    PathConfig()
        : inputBaseDir_("input"),
          outputBaseDir_("output")
    {}

    PathConfig(const fs::path& inputBaseDir, const fs::path& outputBaseDir)
        : inputBaseDir_(inputBaseDir),
          outputBaseDir_(outputBaseDir)
    {}

    fs::path inputCaseDir(const std::string& caseName) const {
        return inputBaseDir_ / caseName;
    }

    fs::path inputCaseDir(const std::string& caseName, const std::string& subCaseName) const {
        return inputBaseDir_ / caseName / subCaseName ;
    }

    fs::path outputCaseDir(const std::string& caseName) const {
        return outputBaseDir_ / caseName;
    }

    fs::path outputCaseDir(const std::string& caseName, const std::string& subCaseName) const{
        return outputBaseDir_ / caseName / subCaseName ;
    }

    fs::path surfaceAFile(const std::string& caseName) const {
        return inputCaseDir(caseName) / "surfaceA.stl";
    }

    fs::path surfaceBFile(const std::string& caseName) const {
        return inputCaseDir(caseName) / "surfaceB.stl";
    }

    fs::path surfaceAPropertiesFile(const std::string& caseName) const {
        return inputCaseDir(caseName) / "surfaceAProperties.csv";
    }

    fs::path surfaceBPropertiesFile(const std::string& caseName) const {
        return inputCaseDir(caseName) / "surfaceBProperties.csv";
    }

    fs::path surfaceAFile(const std::string& caseName, const std::string& subCaseName) const {
        return inputCaseDir(caseName,subCaseName) / "surfaceA.stl";
    }

    fs::path surfaceBFile(const std::string& caseName, const std::string& subCaseName) const {
        return inputCaseDir(caseName,subCaseName) / "surfaceB.stl";
    }

    fs::path surfaceAPropertiesFile(const std::string& caseName, const std::string& subCaseName) const {
        return inputCaseDir(caseName,subCaseName) / "surfaceAProperties.csv";
    }

    fs::path surfaceBPropertiesFile(const std::string& caseName, const std::string& subCaseName) const {
        return inputCaseDir(caseName,subCaseName) / "surfaceBProperties.csv";
    }

    fs::path outputFile(const std::string& caseName, const std::string& filename) const {
        return outputCaseDir(caseName) / filename;
    }

    fs::path outputFile(const std::string& caseName, const std::string& subCaseName, const std::string& filename) const {
        return outputCaseDir(caseName,subCaseName) / filename;
    }

    void validateInputCase(const std::string& caseName) const {
        if (!fs::exists(inputCaseDir(caseName))) {
            throw std::runtime_error(
                "Input case directory not found: " +
                inputCaseDir(caseName).string()
            );
        }

        if (!fs::exists(surfaceAFile(caseName))) {
            throw std::runtime_error("Missing file: " + surfaceAFile(caseName).string());
        }

        if (!fs::exists(surfaceBFile(caseName))) {
            throw std::runtime_error("Missing file: " + surfaceBFile(caseName).string());
        }

        if (!fs::exists(surfaceAPropertiesFile(caseName))) {
            throw std::runtime_error("Missing file: " + surfaceAPropertiesFile(caseName).string());
        }

        if (!fs::exists(surfaceBPropertiesFile(caseName))) {
            throw std::runtime_error("Missing file: " + surfaceBPropertiesFile(caseName).string());
        }
    }

    void validateInputCase(const std::string& caseName, const std::string& subCaseName) const {
        if (!fs::exists(inputCaseDir(caseName, subCaseName))) {
            throw std::runtime_error(
                "Input case directory not found: " +
                inputCaseDir(caseName, subCaseName).string()
            );
        }

        if (!fs::exists(surfaceAFile(caseName, subCaseName))) {
            throw std::runtime_error("Missing file: " + surfaceAFile(caseName, subCaseName).string());
        }

        if (!fs::exists(surfaceBFile(caseName, subCaseName))) {
            throw std::runtime_error("Missing file: " + surfaceBFile(caseName, subCaseName).string());
        }

        if (!fs::exists(surfaceAPropertiesFile(caseName, subCaseName))) {
            throw std::runtime_error("Missing file: " + surfaceAPropertiesFile(caseName, subCaseName).string());
        }

        if (!fs::exists(surfaceBPropertiesFile(caseName, subCaseName))) {
            throw std::runtime_error("Missing file: " + surfaceBPropertiesFile(caseName, subCaseName).string());
        }
    }

    void createOutputCaseDir(const std::string& caseName) const {
        fs::create_directories(outputCaseDir(caseName));
    }

    void createOutputCaseDir(const std::string& caseName, const std::string& subCaseName) const {
        fs::create_directories(outputCaseDir(caseName, subCaseName));
    }
    
};