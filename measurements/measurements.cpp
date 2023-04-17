#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

int main()
{
  std::string path = "./Zameri.txt";
  std::vector<float> mas;
  std::string line;
  std::ifstream in(path); // окрываем файл для чтения
  if (in.is_open())
  {
    while (!in.eof())
    {
      getline(in, line);
      if (line == "") break;
      else 
      {
        mas.push_back(std::stof(line));
      }
    }
  }
  in.close();
  
  std::sort(mas.begin(), mas.end());

  // ================================================================

  std::ofstream out;          // поток для записи
  out.open(path);             // окрываем файл для записи
  float AVG = 0.0;
  if (out.is_open())
  {
    for (auto i : mas)
    {
      AVG += i;
      out << i << std::endl;
    }
    AVG -= mas[0];
    AVG -= mas[mas.size() - 1];
    AVG /= mas.size() - 2;
    out << "\n" << "Average value: " << AVG << "\n";
  }
  out.close();

  mas.clear();
  return 0;
}
