package main

type Layer struct {
	neurons []*Neuron
}

func InitLayer(nin int, nout int, nonLin bool) *Layer {
	neurons := make([]*Neuron, nout)
	for i := range neurons {
		neurons[i] = InitNeuron(nin, nonLin)
	}
	return &Layer{neurons: neurons}
}

func (l *Layer) Forward(x []*Value) []*Value {
	out := make([]*Value, len(l.neurons))
	for i := range l.neurons {
		out[i] = l.neurons[i].Forward(x)
	}
	return out
}

func (l *Layer) Parameters() []*Value {
	var params []*Value
	for _, n := range l.neurons {
		params = append(params, n.Parameters()...)
	}
	return params
}
