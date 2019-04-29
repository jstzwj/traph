#include <algorithm>
#include <memory>

#include <traph/nn/layers/linear.h>
#include <traph/nn/layers/loss.h>
#include <traph/core/tensor.h>
#include <traph/nn/optim.h>

#include <iostream>

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
	
	auto x = traph::ones<traph::f32>({ batch_size,4 });
	auto y = traph::ones<traph::f32>({ batch_size,2 });

	traph::Linear linear_model(4, 2, false);
	traph::MSELoss criterion;
	traph::SGD optimizer(linear_model.parameters(), 0.0001f);
	std::cout << y->data()->to_string() << std::endl;

	std::cout << "Start Training..." << std::endl;

	for (int epoch = 0; epoch < 10000; ++epoch)
	{
		float loss100 = 0.f;

		optimizer.zero_grad();
		auto out = linear_model.forward(x);
		auto loss = criterion.forward(out, y);
		loss->backward();
		optimizer.step();
		// loss100 += loss->item();
		// std::cout << linear_model.parameters()[0]->data()->to_string()<<std::endl;
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
	
    return 0;
}