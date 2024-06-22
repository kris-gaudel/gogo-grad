package main

import (
	"math"
)

type Value struct {
	data      float64
	grad      float64
	_prev     []*Value
	_backward func()
}

func Init(data float64) *Value {
	return &Value{
		data:      data,
		grad:      0.0,
		_prev:     nil,
		_backward: func() {},
	}
}

func (v *Value) Add(other *Value) *Value {
	out := Init(v.data + other.data)
	out._prev = []*Value{v, other}

	out._backward = func() {
		v.grad += out.grad
		other.grad += out.grad
	}
	return out
}

func (v *Value) Mult(other *Value) *Value {
	out := Init(v.data * other.data)
	out._prev = []*Value{v, other}

	out._backward = func() {
		v.grad += other.data * out.grad
		other.grad += v.data * out.grad
	}
	return out
}

func (v *Value) Pow(other *Value) *Value {
	out := Init(math.Pow(v.data, other.data))
	out._prev = []*Value{v, other}

	out._backward = func() {
		v.grad += other.data * math.Pow(v.data, other.data-1) * out.grad
		other.grad += math.Pow(v.data, other.data) * math.Log(v.data) * out.grad
	}
	return out
}

func (v *Value) Relu() *Value {
	out := Init(math.Max(0, v.data))
	out._prev = []*Value{v}

	out._backward = func() {
		if v.data > 0 {
			v.grad += out.grad
		}
	}
	return out
}

func (v *Value) Tanh() *Value {
	out := Init(math.Tanh(v.data))
	out._prev = []*Value{v}

	out._backward = func() {
		v.grad += (1 - out.data*out.data) * out.grad
	}
	return out
}

func (v *Value) Backward() {
	topo := []*Value{}
	visited := map[*Value]bool{}

	var buildTopo func(*Value)
	buildTopo = func(v *Value) {
		if !visited[v] {
			visited[v] = true
			for _, prev := range v._prev {
				buildTopo(prev)
			}
			topo = append(topo, v)
		}
	}
	buildTopo(v)
	v.grad = 1.0
	for i := len(topo) - 1; i >= 0; i-- {
		topo[i]._backward()
	}
}

func (v *Value) Neg() *Value {
	out := v.Mult(Init(-1.0))
	return out
}
