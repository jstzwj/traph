
#include <algorithm>
#include <memory>

#include <traph/nn/layers/linear.h>
#include <traph/nn/layers/loss.h>
#include <traph/core/tensor.h>
#include <traph/tensor/float_tensor.h>
#include <traph/nn/optim.h>

#include <iostream>

using namespace traph;

class MyModel : public Module
{
private:
	std::shared_ptr<Linear> linear1;
	// std::shared_ptr<Linear> linear2;
	// std::shared_ptr<Linear> linear3;
public:

	MyModel()
		:linear1(new Linear(32, 2, false))
		// linear2(new Linear(16, 2, false))
		// linear3(new Linear(256, 128, false))
	{
		add_module("linear1", linear1);
		// add_module("linear2", linear2);
		// add_module("linear3", linear3);
	}
	std::shared_ptr<VariableInterface> forward(std::shared_ptr<VariableInterface> input)
	{
		return linear1->forward(input);
	}
};

int main()
{
	/*
    traph::Tensor<float> a = traph::zeros<float>({4, 3});
    traph::Tensor<float> w = traph::ones<float>({3, 5});
    traph::Tensor<float> result = traph::matmul(a, w);
	*/
	// traph::Tensor<float> result2 = traph::add(a, 1.f);
	/*
	traph::Tensor<traph::f32> a = traph::zeros<traph::f32>({ 5000, 5000 });
	traph::Tensor<traph::f32> b = traph::zeros<traph::f32>({ 5000, 5000 });
	traph::Tensor<traph::f32> c = traph::matmul(a, b);
	*/

	/*
	auto a = traph::Variable<traph::f32>({2, 3});
	auto c = traph::mul(traph::mul(a, a), 3);
	auto out = traph::mean(c);
	out.backward();
	*/
	/*
	traph::Tensor<float> a = traph::ones<float>({ 10000, 10000 });

	auto b = a.sum();
	std::cout << b;
	*/
	// auto a = traph::Variable<traph::f32>({ 2, 3 });
	/*
	auto a = traph::ones<traph::f32>({ 2,3,2 });
	a->requires_grad_(true);
	auto b = traph::sin<traph::f32>(a);
	auto c = traph::ones<traph::f32>({ 2,3,2 });
	c->requires_grad_(true);
	auto d = traph::add<traph::f32>(b, c);
	auto e = traph::sum<traph::f32>(d);

	e->backward();

	std::cout << a->grad()->to_string();
	*/

	/*
	auto a = traph::ones<traph::f32>({ 2,3 });
	a->requires_grad_(true);
	auto b = traph::ones<traph::f32>({ 3,2 });
	b->requires_grad_(true);
	auto c = traph::matmul(a, b);
	auto d = traph::sum(c);
	d->backward();
	std::cout << a->grad()->to_string();
	*/

	int batch_size = 16;
	
	auto x = traph::randn<traph::f32>({ batch_size,32 });
	auto x_data = std::dynamic_pointer_cast<traph::FloatTensor>(x->data());
	// x_data->fill_(0.5f);
	// x_data->data_ptr()[0] = 1.0f;
	std::cout << x_data->to_string() << std::endl;

	auto y = traph::ones<traph::f32>({ batch_size,2 });

	MyModel model;
	// auto param0 = std::dynamic_pointer_cast<Tensor<f32>>(model.parameters()[0]->data());
	// param0->data_ptr()[0] = 0.0001f;
	MSELoss criterion;
	traph::SGD optimizer(model.parameters(), 0.01f);
	// std::cout << y->data()->to_string() << std::endl;

	std::cout << "Start Training..." << std::endl;

	for (int epoch = 0; epoch < 1000; ++epoch)
	{
		float loss100 = 0.f;

		optimizer.zero_grad();
		auto out = model.forward(x);
		auto loss = criterion.forward(out, y);
		loss->backward();
		optimizer.step();
		// std::cout << model.parameters()[0]->grad()->to_string()<<std::endl;
		std::cout << loss->data()->to_string() << std::endl;
	}

	//auto a = traph::ones<traph::f32>({ 2,3 });
	//a->requires_grad_(true);
	//auto b = traph::ones<traph::f32>({ 3,4 });
	//b->requires_grad_(true);
	//auto c = matmul(a, b);
	//auto d = sum(c);
	//d->backward();
	//std::cout << a->grad()->to_string();

	/*
	auto a = std::make_shared<traph::FloatTensor>(DimVector({ 2,1 }));
	auto b = std::make_shared<traph::FloatTensor>(DimVector({ 1,2 }));

	float * a_ptr = a->data_ptr();
	float * b_ptr = b->data_ptr();

	a_ptr[0] = 3;
	a_ptr[1] = 4;

	b_ptr[0] = 5;
	b_ptr[1] = 6;

	auto c = std::dynamic_pointer_cast<traph::FloatTensor>(a->matmul(b));

	std::cout << c->to_string();
	*/
	
    return 0;
}