package dsalgo.practice;

@FunctionalInterface
public interface GenericFunctionInterface<T, R> {
    R func(T t);
}
