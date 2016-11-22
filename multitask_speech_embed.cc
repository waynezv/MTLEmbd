#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/dict.h"
#include "dynet/expr.h"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <iostream>
#include <fstream>

using namespace std;
using namespace dynet;
using namespace dynet::expr;

class Speaker {
  public:
    int speaker_id;
    bool gender; // false = male, true = female
    int age;
    string education;
    string dialect; 
};

class Instance {
  public:
    string input_filename; // name of file containing data for the instance
    string word;
    vector<float> glove_sem_vector;
    Speaker speaker;

    void read_vec();
};

void Instance::read_vec(unordered_map<int, Speaker> speakers_info) {
  string w;
  string speaker_str;
  int speaker_id;
  ifstream in(input_filename);
  {
    getline(in, w);
    word = w;
    getline(in, speaker_str);
    speaker_id = stoi(speaker_str);
    speaker = speakers_info[speaker_id];
  }
}

string strip_string(string s, string to_strip) {
  size_t str_begin = s.find_first_not_of(to_strip);
  size_t str_end = s.find_last_not_of(to_strip);
  size_t str_range = str_end - str_begin + 1;
  if (str_begin == string::npos || str_end == string::npos || str_range <= 0) {
    return "";
  }
  return s.substr(str_begin, str_range);
}

unordered_map<int, Speaker> LoadSpeakers(string speaker_filename) {
  unordered_map<int, Speaker> speakers_info;
  ifstream in(speaker_filename);
  {
    cerr << "Reading speaker data from " << speaker_filename << " ...\n";
    string line;
    string substr;
    int speaker_id;
    bool gender;
    int age;
    string dialect;
    string education;

    while(getline(in, line)) {
      istringstream iss(line);
      vector<string> vect;
      while(iss.good()) {
        getline(iss, substr, ',');
        vect.push_back(substr);
      }

      string speaker_str = strip_string(vect[0], "\" ");
      speaker_id = stoi(speaker_str);

      string gender_str = strip_string(vect[3], "\" ");
      if (gender_str.compare("MALE") == 0) {
        gender = false;
      }
      else {
        gender = true;
      }

      string age_str = strip_string(vect[4], "\" ");
      age = 1997 - stoi(age_str);

      dialect = strip_string(vect[5], "\" ");
      if (dialect.compare("") == 0) {
        dialect = "UNK";
      }

      education = strip_string(vect[6], "\" ");

      Speaker si;
      si.speaker_id = speaker_id;
      si.gender = gender;
      si.age = age;
      si.dialect = dialect;
      si.education = education;
      speakers_info[speaker_id] = si;
    }
  }
  return speakers_info;
}

int main(int argc, char** argv) {
  unordered_map<int, Speaker> speakers_info = LoadSpeakers("../caller_tab.csv");
}