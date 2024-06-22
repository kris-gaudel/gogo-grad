package main

import (
	"math/rand"
	"time"
)

type Neuron struct {
	weights      []*Value
	bias         *Value
	_isNonlinear bool
}

func InitNeuron(nin int, nonLin bool) *Neuron {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	w := make([]*Value, nin)
	for i := range w {
		w[i] = Init(r.Float64()*2 - 1)
	}

	b := Init(0)
	return &Neuron{weights: w, bias: b, _isNonlinear: nonLin}
}

func (n *Neuron) Forward(x []*Value) *Value {
	out := n.bias
	for i := range n.weights {
		out = out.Add(x[i].Mult(n.weights[i]))
	}
	if n._isNonlinear {
		return out.Tanh()
	}
	return out
}

func (n *Neuron) Parameters() []*Value {
	out := []*Value{n.bias}
	out = append(out, n.weights...)
	return out
}
