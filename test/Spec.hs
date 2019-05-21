import           Test.Tasty (TestTree, testGroup)
import qualified Test.Tasty as Tasty
import qualified Test.Tasty.SmallCheck as SC

import           GHC.TypeNats (KnownNat)
import qualified Numeric.LinearAlgebra.Static as LA
import qualified Numeric.LinearAlgebra as LAD
import qualified Data.Map.Strict as M

import qualified Net.Graph as N
import qualified Net.Graph.BiComp as B
import qualified Net.Graph.Marginals as Margs


main :: IO ()
main = Tasty.defaultMain tests


tests :: TestTree
tests = testGroup "Tests" [properties] --, unitTests]


properties :: TestTree
properties = testGroup "Properties" [scProps] -- , qcProps]


scProps = testGroup "(checked by SmallCheck)"
--   [ SC.testProperty "sort == sort . reverse" $
--       \list -> sort (list :: [Int]) == sort (reverse list)
  [ SC.testProperty "rightInTwo (xs ++ xs) == (xs, xs)" $
      \xs -> N.rightInTwo (xs ++ xs :: [Int]) == (xs, xs)
  , Tasty.localOption (SC.SmallCheckDepth 4) .
    SC.testProperty "(decode . encode) x == x" $
      \x -> (N.decode . N.encode) x == x
  , Tasty.localOption (SC.SmallCheckDepth 4) .
    SC.testProperty "(explicate . obfuscate) x == x" $
      \xs ->
        let m = M.fromList (zip N.enumerate $ pad 8 xs)
         in null xs || (N.explicate . N.obfuscate) m == m
  , Tasty.localOption (SC.SmallCheckDepth 4) .
    SC.testProperty "mask (enumerate !! x) ! y == [x == y]" $
      \x0 y0 ->
        let x = x0 `mod` 8
            y = y0 `mod` 8
         in extract y (N.mask (N.enumerate !! x)) ==
              if x == y then 1.0 else 0.0
  , SC.testProperty "explicate (mask x) M.! y == [x == y]" $
      \x y -> N.explicate (B.Vec $ N.mask x) M.! y ==
        if x == y then 1.0 else 0.0
  , Tasty.localOption (SC.SmallCheckDepth 4) .
    SC.testProperty "Res: additive identity" $
      \x -> x + 0 == (x :: Margs.Res Int)
  , Tasty.localOption (SC.SmallCheckDepth 4) .
    SC.testProperty "Res: multiplicative identity" $
      \x -> x * 1 == (x :: Margs.Res Int)
  , Tasty.localOption (SC.SmallCheckDepth 2) .
    SC.testProperty "Res: distributive law" $
      \x y z -> x * (y + z :: Margs.Res Int) == x*y + x*z
  , Tasty.localOption (SC.SmallCheckDepth 2) .
    SC.testProperty "Res: associative multiplication" $
      \x y z -> x * (y * z :: Margs.Res Int) == (x * y) * z
  ]


-- | Pad the given (non-empty) list to the given number of elements.
pad :: Int -> [a] -> [a]
pad k = take k . cycle


-- | Extract the `k`-th element of `R d` vector.
--
-- TODO: maybe could be moved to some utility module?
--
extract :: (KnownNat n) => Int -> LA.R n -> Double
extract k v = (LAD.toList . LA.unwrap) v !! k
