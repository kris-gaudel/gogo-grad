package main

func SGD(model *MLP, index int, iters int) {
	learningRate := 1.0 - 0.9*float64(index)/float64(iters)
	for _, p := range model.Parameters() {
		delta := learningRate * p.grad
		p.data -= delta
	}
}
