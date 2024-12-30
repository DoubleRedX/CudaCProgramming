//
// Created by Gary27 on 2024/12/30.
//

#include <iostream>
#include <cstdlib>


int main(){

  const char *command = "ls -l"; // 要执行的命令
  char buffer[128];

  // 打开命令管道
  FILE* pipe = popen(command, "r");
  if (!pipe) {
    std::cerr << "Failed to run command!" << std::endl;
    return -1;
  }

  // 读取命令输出并打印
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    std::cout << buffer;
  }

  // 关闭管道
  int returnCode = pclose(pipe);
  std::cout << "\nCommand executed with return code: " << returnCode << std::endl;

  return 0;
}