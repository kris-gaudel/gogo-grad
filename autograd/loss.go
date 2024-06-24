package main

func Loss(model *MLP, xs [][]float64, ys []float64) (*Value, float64) {
	// Convert inputs to Value types
	inputs := make([][]*Value, len(xs))
	for i, xrow := range xs {
		inputs[i] = []*Value{Init(xrow[0]), Init(xrow[1])}
	}

	// Forward pass
	scores := make([]*Value, len(inputs))
	for i, xrow := range inputs {
		scores[i] = model.Forward(xrow)[0]
	}

	// Computing Support Vector Machine max-margin loss
	losses := make([]*Value, len(ys))
	for i, yi := range ys {
		losses[i] = Init(1.0).Add(Init(-yi).Mult(scores[i])).Relu()
	}
	n := float64(len(losses))
	dataLoss := losses[0]
	for _, loss := range losses[1:] {
		dataLoss = dataLoss.Add(loss)
	}
	dataLoss = dataLoss.Mult(Init(1.0 / n))

	// L2 Regularization
	alpha := 0.0001
	regLoss := Init(0.0)
	for _, param := range model.Parameters() {
		regLoss = regLoss.Add(param.Mult(param))
	}
	regLoss = regLoss.Mult(Init(alpha))
	totalLoss := dataLoss.Add(regLoss)

	// Compute Accuracy
	correct := 0
	for i, yi := range ys {
		if (yi > 0) == (scores[i].data > 0) {
			correct++
		}
	}
	accuracy := float64(correct) / n

	return totalLoss, accuracy
}
