#pragma once
#include <iostream>
#include <string>
#include <sys/stat.h>
namespace DATA {

template<typename T>
class BinFile {
public:
    BinFile(int split_bast_set_rows): split_bast_set_rows(split_bast_set_rows) {
        if (base_set_fp_) fclose(base_set_fp_);
        if (query_set_fp_) fclose(query_set_fp_);
        base_set_fp_ = nullptr;
        query_set_fp_ = nullptr;
    }
    
    void open_file(std::string file_name, bool is_base_set) {
        FILE* fp_ = nullptr;
        fp_ = fopen(file_name.c_str(), "r"); //返回文件指针
        struct stat statbuf;
        if (stat(file_name.c_str(), &statbuf) != 0) { throw std::runtime_error("stat() failed: " + file_name); }
        file_size = statbuf.st_size;

        uint32_t header[2];
        if (fread(header, sizeof(uint32_t), 2, fp_) != 2) {
            throw std::runtime_error("read header of BinFile failed: " + file_name);
        }
        uint32_t nrows = header[0]; 
        uint32_t ndims = header[1];
        //std::cout<< " ====== file_size_ is : " << file_size << std::endl;
        //std::cout<< " ====== nrows_ is : " << nrows << " ====== ndims_ is : " << ndims << std::endl;
        //size_t expected_file_size = 2 * sizeof(uint32_t) + static_cast<size_t>(nrows) * ndims* sizeof(T);
        /*
        if (file_size != expected_file_size) {
            throw std::runtime_error("expected file size of " + file_name + " is " +
                               std::to_string(expected_file_size) + ", however, actual size is " +
                               std::to_string(file_size));
        }
        */
        if (is_base_set) {
            base_set_fp_ = fp_;
            base_set_nrows = (split_bast_set_rows == 0) ? nrows : split_bast_set_rows;
            base_set_ndims = ndims;
            std::cout << " ===== base_set_nrows is : " << base_set_nrows << " base_set_ndims is : " << base_set_ndims << std::endl;
        } else {
            query_set_fp_ = fp_;
            query_set_nrows = nrows;
            query_set_ndims = ndims;
            std::cout << " ===== query_set_nrows is : " << query_set_nrows << " query_set_ndims is : " << query_set_ndims << std::endl;
        }
        
    }

    void read_base_set() {
        
        base_set = new T[base_set_nrows * base_set_ndims];
        size_t total = static_cast<size_t>(base_set_nrows) * base_set_ndims;
        //size_t  true_total = fread(base_set, sizeof(T), total, base_set_fp_);
        //std::cout << " ====== true_total is : " << true_total << std::endl; 
        if (fread(base_set, sizeof(T), total, base_set_fp_) != total) {
            throw std::runtime_error("fread() BinFile Base Set failed");
        }
        fclose(base_set_fp_);
    }

    void read_query_set() {
        query_set = new T[query_set_nrows * query_set_ndims];
        size_t total = static_cast<size_t>(query_set_nrows) * query_set_ndims;
        if (fread(query_set, sizeof(T), total, query_set_fp_) != total) {
            throw std::runtime_error("fread() BinFile Query Set failed");
        }
        fclose(query_set_fp_);
    }

public:
    size_t base_set_nrows;
    size_t base_set_ndims;
    size_t query_set_nrows;
    size_t query_set_ndims;
    FILE* base_set_fp_ = nullptr;
    FILE* query_set_fp_ = nullptr;
    size_t file_size;
    T* base_set =nullptr;
    T* query_set = nullptr;

    int split_bast_set_rows = 0;
};
} //end namespace DATA