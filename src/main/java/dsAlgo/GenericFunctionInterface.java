package dsAlgo;

@FunctionalInterface
public interface GenericFunctionInterface<T, R> {
    R func(T t);
}
