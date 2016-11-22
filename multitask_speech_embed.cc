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

bool TO_TRAIN = true;
string model_name = "model.model";
int EMBEDDING_SIZE = 1024;
int CONV_OUTPUT_DIM = 1024;
int GLOVE_DIM = 50;
int CONV1_FILTERS = 10;
int CONV1_SIZE = 8;
int CONV2_FILTERS = 10;
int CONV2_SIZE = 10;
int ROWS1 = 40;
int ROWS2 = 4;

Dict speaker_d;
Dict education_d;
Dict dialect_d;
Dict word_d;

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
    vector<float> input_vector;
    Speaker speaker;

    void read_vec(unordered_map<int, Speaker> speakers_info);
};

enum Task { WORD, SEM_SIMILARITY, SPEAKER_ID, GENDER, AGE, EDUCATION, DIALECT };

void Instance::read_vec(unordered_map<int, Speaker> speakers_info) {
  string w;
  string speaker_str;
  int speaker_id;
  string line;
  string substr;
  ifstream in(input_filename);
  {
    getline(in, w);
    word = w;
    getline(in, speaker_str);
    speaker_id = stoi(speaker_str);
    speaker = speakers_info[speaker_id];
    vector<float> instance_vector; 
    while(getline(in, line)) {
      istringstream iss(line);
      while(iss.good()) {
        getline(iss, substr, ',');
        instance_vector.push_back(stof(substr));
      }
    }
    input_vector = instance_vector;
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

unordered_map<int, Speaker> load_speakers(string speaker_filename) {
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

      speaker_d.convert(speaker_id);
      dialect_d. convert(dialect);
      education_d.convert(education);

      speakers_info[speaker_id] = si;
    }
  }
  return speakers_info;
}

Struct MTLBuilder {
  vector<Parameter> p_ifilts; //filters for the 1dconv over the input
  vector<Parameter> p_cfilts; //filters for the 1dconv over the (altered) output of the first convolution
  Parameter p_c2we; //the output of the convolution to the word embedding
  Parameter p_we2sr; //the word embedding to speech recognition
  Parameter p_we2ss; //word embedding to the semantic similarity
  Parameter p_we2id; //word embedding to the speaker id
  Parameter p_we2gen; //word embedding to gender
  Parameter p_we2age; //word embedding to age
  Parameter p_we2edu; //word embedding to education
  Parameter p_we2dia;  //word embedding to dialect

  explicit MTLBuilder(Model* model) :
    p_c2we(model->add_parameters({EMBEDDING_SIZE, CONV_OUTPUT_DIM})),
    p_we2sr(model->add_parameters({word_d.size(), EMBEDDING_SIZE})),
    p_we2ss(model->add_parameters({GLOVE_DIM, EMBEDDING_SIZE})),
    p_we2id(model->add_parameters({speaker_d.size(), EMBEDDING_SIZE})),
    p_we2gen(model->add_parameters({2, EMBEDDING_SIZE})),
    p_we2age(model->add_parameters({1, EMBEDDING_SIZE})),
    p_we2edu(model->add_parameters({education_d.size(), EMBEDDING_SIZE})),
    p_we2dia(model->add_parameters({dialect_d.size(), EMBEDDING_SIZE})) {
      for (int i = 0; i < CONV1_SIZE; ++i) {
        p_ifilts.push_back(model->add_parameters({ROWS1, CONV1_SIZE}));
      }
      for (int i = 0; i < CONV2_SIZE; ++i) {
        p_cfilts.push_back(model->add_parameters({ROWS2, CONV1_SIZE}));

      }
    }

    Expression loss_against_task(ComputationGraph& cg, Task task, Instance instance) {
      Expression raw_input = input(cg, {instance.input_vector.size()/40, 40}, input_vector);
      Expression input = transpose(raw_input);

      vector<Expression> conv1_out;
      for (int i = 0; i < CONV1_SIZE; ++i) {
        conv1_out.push_back(conv1d_wide(input, parameter(cg, p_ifilts[i])));
      }      

    }
}




int main(int argc, char** argv) {
  unordered_map<int, Speaker> speakers_info = LoadSpeakers("../caller_tab.csv");
  dynet::initialize(argc, argv, 3640753077);




  unordered_map<int, Speaker> speakers_info = load_speakers("../caller_tab.csv");
  speaker_d.freeze();
  education_d.freeze();
  dialect_d.freeze();
  word_d.freeze();

  speaker_d.set_unk("UNK");
  education_d.set_unk("UNK");
  dialect_d.set_unk("UNK");
  word_d.set_unk("UNK");

  vector<int> order;
  for (int i = 0; i < instances.size(); ++i) {
    order.push_back(i);
  }

  int iter = -1;
  if (TO_TRAIN) {
    ++iter;
    while(true) {
      

    }

  }
}
