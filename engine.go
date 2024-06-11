package main

type Value struct {
	Data  float64
	_prev []*Value
	_op   string
}

func InitValue(data float64) *Value {
	return NewValue(data, nil, "")
}

func NewValue(data float64, prev []*Value, op string) *Value {
	return &Value{Data: data, _prev: prev, _op: op}
}

func (v *Value) Add(other *Value) *Value {
	return NewValue(v.Data+other.Data, []*Value{v, other}, "+")
}

func (v *Value) Mult(other *Value) *Value {
	return NewValue(v.Data*other.Data, []*Value{v, other}, "*")
}
