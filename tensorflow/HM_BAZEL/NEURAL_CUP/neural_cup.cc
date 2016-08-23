
#include "neural_cup.h"
using namespace tensorflow;
using namespace std;
bool neural_cup(float* readinput) {

  // Initialize a tensorflow session
  Session* session;
  Status status = NewSession(SessionOptions(), &session);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 0;
  }

  // Read in the protobuf graph we exported
  // (The path seems to be relative to the cwd. Keep this in mind
  // when using `bazel run` since the cwd isn't where you call
  // `bazel run` but from inside a temp folder.)
  GraphDef graph_def;
  status = ReadBinaryProto(Env::Default(), "models/ret2.pb", &graph_def);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 0;
  }

  // Add the graph to the session
  status = session->Create(graph_def);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 0;
  }

  // Setup inputs and outputs:

  // Our graph doesn't require any inputs, since it specifies default values,
  // but we'll change an input to demonstrate.
 
  Tensor x(DT_FLOAT, TensorShape({1,14}));
  for(int s=0;s<14;s++)
  {
    x.matrix<float>()(0,s) = *(readinput + s);
  }

  std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
    { "inputx", x },
  };

  // The session will initialize the outputs
  std::vector<tensorflow::Tensor> outputs;
  Tensor checkpoint_filepath(DT_STRING, TensorShape());
  checkpoint_filepath.scalar<std::string>()() = "models/recordCUPDATA2";
  session->Run( {{ "save/Const", checkpoint_filepath },}, 
                       {}, {"save/restore_all"}, &outputs);
  
  //std::vector<tensorflow::Tensor> outputs;
  // Run the session, evaluating our "c" operation from the graph
  session->Run(inputs, {"add_4"}, {}, &outputs);
  

  // Grab the first output (we only evaluated one graph node: "c")
  // and convert the node to a scalar representation.
  //auto output_c = outputs[0].scalar<float>();
  vector<Tensor>::iterator it = outputs.begin();
  auto items = it->shaped<float, 2>({1, 2});
  std::cout<<items(0,0)<<"\n";
  std::cout<<items(0,1)<<"\n";
  if (items(0,0)>items(0,1))
  {
    session->Close();
    return 0;
  }
  else
  {
    session->Close();
    return 1;
  }
  // (There are similar methods for vectors and matrices here:
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/tensor.h)

  // Print the results
  //std::cout << outputs[0].DebugString() << "\n"; // Tensor<type: float shape: [] values: 30>
  //std::cout << output_c() << "\n"; // 30

  // Free any resources used by the session
  
  return 0;
}
