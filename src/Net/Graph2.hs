{-# LANGUAGE CPP #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE DeriveFunctor #-}

{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}

{-# LANGUAGE PolyKinds #-}

{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE Rank2Types #-}

{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeFamilies #-}

-- To automatically derive Binary
{-# LANGUAGE DeriveAnyClass #-}


-- | Processing graphs with artificial neural networks.  This module allows to
-- assign labels to both arcs (as in `Net.Graph`) and nodes (in contrast to
-- `Net.Graph`).
--
-- The plan is to play with the arc-factored model first.  Two inference
-- methods can be considered: let's call them local and global.  In either
-- case, we assign potentials two combinations of values:
--
--   * (label of the head, label of the arc, label of the dependent)
--
-- In local inference, we split this assignment to make the potentials of:
--
--   * label of the head
--   * label of the arc
--   * label of the dependent
--
-- completely separate.  Then we can simply sum the potentials before making
-- the final decision as to whether a given node should be considered an MWE
-- component or not.  Again, this makes the decisions for the individual nodes
-- and arcs independent (at least intuitively, I'm not sure if formally these
-- decisions are completely independent).
--
-- In case of global inference, we want to maximize the total potential of a
-- labeling of the whole dependency tree.  We can certainly do that at the time
-- of tagging, not sure how/if this could work during training.


module Net.Graph2
  ( 
  -- * Network
    new
  , run
  , eval
  , evalRaw
  , ProbTyp(..)

  -- * Opaque
  , Opaque(..)
  , Typ(..)
  , newO

  -- * Transparent
  , Transparent(..)
  , inpMod
  , traMod
  , biaMod
  , netErrorT

  -- * Serialization
  , save
  , load

  -- * Data set
  , Elem(..)
  , tokens
  , replace

  -- * Error
  , netError

  -- * Encoding
  , Out(..)
  , encode
  , decode
  , rightInTwo

  -- * Explicating
  , B.enumerate
  , explicate
  , obfuscate
  , B.mask

  -- * Inference
  , Labeling(..)
  , tagGreedy
  , treeTagGlobal
  , treeTagConstrained

  -- * Trees
  , treeConnectAll
  ) where


import           GHC.Generics (Generic)
import           GHC.TypeNats (KnownNat, natVal)
-- import           GHC.Natural (naturalToInt)
import qualified GHC.TypeNats as Nats
import           GHC.TypeLits (Symbol, KnownSymbol)

import           System.Random (randomRIO)

import           Control.Monad (forM_, forM, guard)

import           Lens.Micro.TH (makeLenses)
-- import           Lens.Micro ((^.))

-- import           Data.Monoid (Sum(..))
import           Data.Monoid (Any(..))
import           Data.Proxy (Proxy(..))
-- import           Data.Ord (comparing)
import qualified Data.List as List
import qualified Data.Foldable as F
import           Data.Maybe (catMaybes, mapMaybe)
import qualified Data.Text as T
import qualified Data.Set as S
import qualified Data.Map.Strict as M
import qualified Data.Graph as G
import qualified Data.Array as A
import           Data.Binary (Binary)
import qualified Data.Binary as Bin
import qualified Data.ByteString.Lazy as BL
import qualified Data.Vector.Sized as VS
import           Codec.Compression.Zlib (compress, decompress)

-- import qualified Data.Number.LogFloat as LF
-- import           Data.Number.LogFloat (LogFloat)

-- import qualified Text.Read as R
-- import qualified Text.ParserCombinators.ReadP as R

-- import qualified Data.Map.Lens as ML
-- import           Control.Lens.At (ixAt)
import           Control.Lens.At (ix)
import           Control.Lens (Lens)
import           Control.DeepSeq (NFData)

import           Dhall (Interpret)
import qualified Data.Aeson as JSON

import qualified Prelude.Backprop as PB
import qualified Numeric.Backprop as BP
import           Numeric.Backprop (Backprop, (^^.), (^^?))
import qualified Numeric.LinearAlgebra.Static.Backprop as LBP
import           Numeric.LinearAlgebra.Static.Backprop
  (R, L, BVar, Reifies, W, (#), (#>), dot)
import qualified Numeric.LinearAlgebra.Static as LA

import           Net.New
import           Net.Pair
import           Net.Util
-- import qualified Net.List as NL
import qualified Net.FeedForward as FFN
import           Net.FeedForward (FFN(..))
-- import qualified GradientDescent as GD
-- import qualified GradientDescent.Momentum as Mom
-- import qualified GradientDescent.Nestorov as Mom
-- import qualified GradientDescent.AdaDelta as Ada
import           Numeric.SGD.ParamSet (ParamSet)

import qualified Format.Cupt as Cupt

import           Graph
import qualified Net.List as NL
import           Net.Graph2.BiComp
  (Pot, Prob, Vec(..), Vec8, Out(..))
import qualified Net.Graph2.BiComp as B
import qualified Net.Graph2.UniComp as U
import qualified Net.Graph2.Marginals as Margs
import qualified Net.Input as I

import           Debug.Trace (trace)


----------------------------------------------
-- Transparent
----------------------------------------------


-- | Should be rather named sth. like `Fixed`...
data Transparent = Transparent
  { _inpMod :: I.PosDepInp 25 25
  -- , _traMod :: I.ScaleLeakyRelu 350 150
  , _traMod :: I.Scale 350 150
  , _uniMod :: U.UniAff 150 100
  , _biaMod :: B.BiAff 150 100
  } deriving (Generic, Binary, NFData, ParamSet, Backprop)

-- data Transparent = Transparent
--   { _inpMod :: I.RawInp
--   , _biaMod :: B.BiAff 300 100
--   } deriving (Generic, Binary, NFData, ParamSet, Backprop)

makeLenses ''Transparent

instance (a ~ T.Text, b ~ T.Text) => New a b Transparent where
  new xs ys = Transparent
    <$> new xs ys <*> new xs ys <*> new xs ys <*> new xs ys


-- | Net error with `Transparent`
netErrorT
  :: (Reifies s W)
  => ProbTyp
  -> Elem (R 300)
  -> BVar s Transparent
  -> BVar s Double
netErrorT ptyp x net =
  -- TODO: maybe should rely on `Sent`, as defined in the `MWE2` module?  Also
  -- have a look at `tagT` in `MWE2`.
  let toksEmbs = tokens x
      embs' = I.runTransform (net ^^. traMod) 
            . I.runInput (net ^^. inpMod) 
            $ toksEmbs
      x' = replace embs' x
   in netError' ptyp [x'] (net ^^. biaMod) (net ^^. uniMod)


----------------------------------------------
-- Opaque2
----------------------------------------------

-- -- | Each constructor of `ITyp` corresponds to a specific feature extraction
-- -- architecture.  The parameter @i@ corresponds to the size of the output.
-- data ITyp = PosDep
--   deriving (Generic, Binary, Read)
-- 
-- 
-- data Opaque2 :: Nats.Nat -> * -> * -> * where
--   Opaque2 
--     :: ( Binary p, NFData p, ParamSet p, I.Input p d 350
--        , Binary q, NFData q, ParamSet q, B.BiComp 350 q
--        )
--     => ITyp -> p
--     -> Typ -> q
--     -> Opaque2 d a b
-- 
-- 
-- instance (KnownNat d) => Binary (Opaque2 d a b) where
--   put (Opaque2 typ p ityp q) = do
--     Bin.put ityp
--     Bin.put q
--     Bin.put typ
--     Bin.put p
--   get = do
--     typ <- Bin.get
--     p <- case typ of
--       Arc0T -> Bin.get :: Bin.Get (Arc0 d)
-- --       Arc1T -> Opaque typ <$> (action :: m (Arc1 d))
-- --       Arc2T -> Opaque typ <$> (action :: m (Arc2 d))
-- --       Arc3T -> Opaque typ <$> (action :: m (Arc3 d))
--     ityp <- Bin.get
--     q <- case ityp of
--       PosDep -> Bin.get :: Bin.Get (I.PosDepInp 25 25)
--     return $ Opaque2 ityp q typ p
-- --     ityp <- Bin.get


----------------------------------------------
-- Opaque
----------------------------------------------


-- | Each constructor of `Typ` corresponds to a specific network architecture.
data Typ
  = Arc0T
  | Arc1T
  | Arc2T
  | Arc3T
  deriving (Generic, Binary, Read)


-- | Opaque neural network with the actual architecture abstracted away.
-- The type (`Typ`) is kept for the sake of (de-)serialization.
--
-- `Opaque` is motivated by the fact that each network architecture is
-- represeted by a different type.  We know, however, that each architecture is
-- an instance of `B.QuadComp`, `ParamSet`, etc., and this is precisely
-- what `Opaque` preserves.
data Opaque :: Nats.Nat -> * -> * -> * where
  Opaque 
    :: (Binary p, NFData p, ParamSet p, B.BiComp d p)
    => Typ -> p -> Opaque d a b


instance (KnownNat d) => Binary (Opaque d a b) where
  put (Opaque typ p) = Bin.put typ >> Bin.put p
  get = do
    typ <- Bin.get
    exec typ Bin.get


-- | Execute the action within the context of the given model type.  Just a
-- helper function, really, which allows to avoid boilerplate code.
exec
  :: forall d a b m. (KnownNat d, Functor m)
  => Typ
  -> (forall p. (Binary p, New a b p) => m p)
  -> m (Opaque d a b)
exec typ action =
  case typ of
    Arc0T -> Opaque typ <$> (action :: m (Arc0 d))
    Arc1T -> Opaque typ <$> (action :: m (Arc1 d))
    Arc2T -> Opaque typ <$> (action :: m (Arc2 d))
    Arc3T -> Opaque typ <$> (action :: m (Arc3 d))


-- | Create a new network of the given type.
newO
  :: forall d a b.
    ( KnownNat d
--     , Binary a, NFData a, Show a, Ord a
--     , Binary b, NFData b, Show b, Ord b
    )
  => Typ     -- ^ Network type, which determines its architecture
  -> S.Set a -- ^ The set of node labels
  -> S.Set b -- ^ The set of arc labels
  -> IO (Opaque d a b)
newO typ xs ys =
  exec typ (new xs ys)


----------------------------------------------
-- Different network architectures
----------------------------------------------


-- | Arc-factored model (0)
type Arc0 d
   = B.BiAff d 100


-- | Arc-factored model (1)
type Arc1 d
   = B.BiAff d 200


-- | Version of `Arc7` with mixed joint and independent potential calculation.
type Arc2 d
   = B.BiAffMix d 200


-- | `Arc8` with increased hidden layer (200 -> 400)
type Arc3 d
   = B.BiAffMix d 400


----------------------------------------------
-- Evaluation
----------------------------------------------


-- | Typ of probabilities to employ
data ProbTyp
  = SoftMax
    -- ^ Just softmax
  | Marginals -- Int
    -- ^ Marginals (approximated)
  | Constrained -- Int
    -- ^ Constrained version of marginals
  deriving (Generic)

instance Interpret ProbTyp


-- | Run the given (bi-affine) network over the given graph within the context
-- of back-propagation.
runRaw
  :: ( KnownNat dim --, Ord a, Show a, Ord b, Show b
     , Reifies s W
     , B.BiComp dim comp
     )
  => BVar s comp
    -- ^ Graph network / params
  -> Graph (BVar s (R dim)) ()
    -- ^ Input graph labeled with initial hidden states
  -> M.Map Arc (BVar s (Vec8 Pot))
    -- ^ Output map with output potential values
runRaw net graph = M.fromList $ do
  arc <- graphArcs graph
  let x = B.runBiComp graph arc net
  return (arc, x)


-- | Run the given uni-affine network over the given graph within the context
-- of back-propagation.
runRawUni
  :: ( KnownNat dim
     , Reifies s W
     , U.UniComp dim comp
     )
  => BVar s comp
    -- ^ Graph network / params
  -> Graph (BVar s (R dim)) ()
    -- ^ Input graph labeled with hidden states
  -> M.Map G.Vertex (BVar s Double)
    -- ^ Output map with output potential values
runRawUni net graph = M.fromList $ do
  v <- graphNodes graph
  let x = U.runUniComp graph v net
  return (v, x)


-- | `runRaw` + softmax / approximate marginals
run
  :: ( KnownNat dim -- , Ord a, Show a, Ord b, Show b
     , Reifies s W
     , B.BiComp dim comp
     )
  => ProbTyp
  -> BVar s comp
    -- ^ Graph network / params
  -> Graph (BVar s (R dim)) ()
    -- ^ Input graph labeled with initial hidden states
  -> M.Map Arc (BVar s (Vec8 Prob))
    -- ^ Output map with output potential values
run probTyp net graph =
  case probTyp of
    SoftMax -> fmap B.softmaxVec (runRaw net graph)
    Marginals -> Margs.approxMarginals graph (runRaw net graph) 1
    Constrained -> Margs.approxMarginalsC graph (runRaw net graph) 1


-- | `runRaw` + `runRawUni` + softmax / approximate marginals
runBoth
  :: ( KnownNat dim -- , Ord a, Show a, Ord b, Show b
     , Reifies s W
     , B.BiComp dim comp
     , U.UniComp dim comp'
     )
  => ProbTyp
  -> BVar s comp
  -> BVar s comp'
  -> Graph (BVar s (R dim)) ()
    -- ^ Input graph labeled with initial hidden states
  -> M.Map Arc (BVar s (Vec8 Prob))
    -- ^ Output map with output potential values
runBoth probTyp net netU graph =
  case probTyp of
    SoftMax -> error "runBoth: softmax not implemented"
    Marginals ->
      Margs.approxMarginals' graph
        (runRaw net graph) 
        (runRawUni netU graph)
        1
    Constrained -> error "runBoth: constrained not implemented"


-- | Evaluate the network over the given graph.  User-friendly (and without
-- back-propagation) version of `runRaw`.
evalRaw
  :: ( KnownNat dim -- , Ord a, Show a, Ord b, Show b
     , B.BiComp dim comp
     )
  => comp
    -- ^ Graph network / params
  -> Graph (R dim) ()
    -- ^ Input graph labeled with initial hidden states (word embeddings)
  -> M.Map Arc (Vec8 Pot)
evalRaw net graph =
  BP.evalBP0
    ( BP.collectVar
    $ runRaw
        (BP.constVar net)
        (nmap BP.auto graph)
    )


-- | Evaluate the network over the given graph.  User-friendly (and without
-- back-propagation) version of `run`.
--
-- TODO: lot's of code common with `evalRaw`.
--
eval
  :: ( KnownNat dim --, Ord a, Show a, Ord b, Show b
     , B.BiComp dim comp
     )
  => ProbTyp
  -> comp
    -- ^ Graph network / params
  -> Graph (R dim) ()
    -- ^ Input graph labeled with initial hidden states (word embeddings)
  -> M.Map Arc (Vec8 Prob)
eval probTyp net graph =
  BP.evalBP0
    ( BP.collectVar
    $ run probTyp
        (BP.constVar net)
        (nmap BP.auto graph)
    )


----------------------------------------------
-- Merging
----------------------------------------------


-- -- | Merge results from affinity and biaffinity components.
-- merge
--   :: M.Map G.Vertex (BVar s Double)
--     -- ^ Node potentials
--   -> M.Map Arc (BVar s (Vec8 Pot))
--     -- ^ Arc potentials
--   -> M.Map Arc (BVar s (Vec8 Pot))
-- merge nodPotMap arcPotMap = M.fromList $ do
--   ((v, w), v8) <- M.toList arcPotMap


----------------------------------------------
----------------------------------------------
-- Inference/decoding/tagging
----------------------------------------------
----------------------------------------------


-- | Graph node/arc labeling
data Labeling a = Labeling
  { nodLab :: M.Map G.Vertex a
  , arcLab :: M.Map Arc a
  } deriving (Functor)

instance Semigroup a => Semigroup (Labeling a) where
  Labeling n1 a1 <> Labeling n2 a2 =
    Labeling (n1 <> n2) (a1 <> a2)

instance Monoid a => Monoid (Labeling a) where
  mempty = Labeling M.empty M.empty


----------------------------------------------
-- Greedy decoding
----------------------------------------------


-- | Collect the vectorized probabilities over arcs and nodes.
collect :: M.Map Arc (Vec8 Prob) -> Labeling [Double]
collect =
  mconcat . map spreadOne . M.toList
  where
    spreadOne ((v, w), vec) =
      let Out{..} = decode vec
       in Labeling
            { nodLab = M.fromList [(v, [depVal]), (w, [hedVal])]
            , arcLab = M.singleton (v, w) [arcVal]
            }


-- | Greedily pick the labeling with high potential based on the given
-- potential map and the given MWE choice function.
tagGreedy
  :: ([Double] -> Bool)
    -- ^ MWE choice for a given list of probabilities, independently assigned
    -- to a given object (arc or node)
  -> M.Map Arc (Vec8 Pot)
  -> Labeling Bool
tagGreedy mweChoice =
  let softMax vec = BP.evalBP0 $ B.softmaxVec (BP.auto vec)
   in fmap mweChoice . collect . fmap softMax


----------------------------------------------
-- Global decoding'
----------------------------------------------


-- | Determine the node/arc labeling which maximizes the global potential over
-- the given tree and return the resulting arc labeling.
--
-- WARNING: This function is only guaranteed to work correctly if the argument
-- graph is a tree!
--
treeTagGlobal'
  :: Graph a b
  -> M.Map Arc (Vec8 Pot)
  -> M.Map G.Vertex Double
  -> Labeling Bool
treeTagGlobal' graph labMap nodMap =
  let (trueBest, falseBest) =
        tagSubTree'
          (treeRoot graph)
          graph
          (fmap explicate labMap)
          nodMap
      best = better trueBest falseBest
   in fmap getAny (bestLab best)


-- | The function returns two `Best`s:
--
--   * The best labeling if the label of the root is `True`
--   * The best labeling if the label of the root is `False`
--
tagSubTree'
  :: G.Vertex
    -- ^ Root of the subtree
  -> Graph a b
    -- ^ The entire graph
  -> M.Map Arc (M.Map (Out Bool) Double)
    -- ^ Explicated labeling potentials
  -> M.Map G.Vertex Double
    -- ^ Node labeling potentials
  -> (Best, Best)
tagSubTree' root graph lmap nmap =
  (bestWith True, bestWith False)
  where
    nodePot rootVal
      | rootVal = nmap M.! root
      | otherwise = 0.0
    bestWith rootVal = addNode root rootVal (nodePot rootVal) . mconcat $ do
      child <- Graph.incoming root graph
      let arc = (child, root)
          pot arcv depv = (lmap M.! arc) M.!
            Out {arcVal=arcv, hedVal=rootVal, depVal=depv}
          (true, false) = tagSubTree' child graph lmap nmap
      return $ List.foldl1' better
        [ addArc arc True  (pot True  True)  true
        , addArc arc False (pot False True)  true
        , addArc arc True  (pot True  False) false
        , addArc arc False (pot False False) false ]


----------------------------------------------
-- Global decoding
----------------------------------------------


-- | Determine the node/arc labeling which maximizes the global potential over
-- the given tree and return the resulting arc labeling.
--
-- WARNING: This function is only guaranteed to work correctly if the argument
-- graph is a tree!
--
treeTagGlobal
  :: Graph a b
  -> M.Map Arc (Vec8 Pot)
  -> Labeling Bool
treeTagGlobal graph labMap =
  let (trueBest, falseBest) =
        tagSubTree
          (treeRoot graph)
          graph
          (fmap explicate labMap)
      best = better trueBest falseBest
   in fmap getAny (bestLab best)


-- | The best arc labeling for a given subtree.
data Best = Best
  { bestLab :: Labeling Any
    -- ^ Labeling (using `Any` guarantees that disjunction is used in case some
    -- label is accidentally assigned to a given object twice)
  , bestPot :: Double
    -- ^ Total potential
  }

instance Semigroup Best where
  Best l1 p1 <> Best l2 p2 =
    Best (l1 <> l2) (p1 + p2)

instance Monoid Best where
  mempty = Best mempty 0


-- | Impossible labeling (with infinitely poor potential)
impossible :: Best
impossible = Best mempty (read "-Infinity")


-- | Choose the better `Best`.
better :: Best -> Best -> Best
better b1 b2
  | bestPot b1 >= bestPot b2 = b1
  | otherwise = b2


-- | Add the given arc, its labeling, and the resulting potential to the given
-- `Best` structure.
addArc :: Arc -> Bool -> Double -> Best -> Best
addArc arc lab pot Best{..} = Best
  { bestLab = bestLab
      {arcLab = M.insert arc (Any lab) (arcLab bestLab)}
  , bestPot = bestPot + pot 
  }


-- | Set label of the given node in the given `Best` structure.  Similar to
-- `addArc`, but used when the potential of the node has been already accounted
-- for.
setNode :: G.Vertex -> Bool -> Best -> Best
setNode node lab best@Best{..} = best
  { bestLab = bestLab
      {nodLab = M.insert node (Any lab) (nodLab bestLab)}
  }


-- | Add the given node, its labeling, and the resulting potential to the given
-- `Best` structure.
addNode :: G.Vertex -> Bool -> Double -> Best -> Best
addNode node lab pot Best{..} = Best
  { bestLab = bestLab
      {nodLab = M.insert node (Any lab) (nodLab bestLab)}
  , bestPot = bestPot + pot
  }


-- | The function returns two `Best`s:
--
--   * The best labeling if the label of the root is `True`
--   * The best labeling if the label of the root is `False`
--
tagSubTree
  :: G.Vertex
    -- ^ Root of the subtree
  -> Graph a b
    -- ^ The entire graph
  -- -> M.Map Arc (Vec8 Prob)
  -> M.Map Arc (M.Map (Out Bool) Double)
    -- ^ Explicated labeling potentials
  -> (Best, Best)
tagSubTree root graph lmap =
  (bestWith True, bestWith False)
  where
    bestWith rootVal = setNode root rootVal . mconcat $ do
      child <- Graph.incoming root graph
      let arc = (child, root)
          pot arcv depv = (lmap M.! arc) M.!
            Out {arcVal=arcv, hedVal=rootVal, depVal=depv}
          (true, false) = tagSubTree child graph lmap
      return $ List.foldl1' better
        [ addArc arc True  (pot True  True)  true
        , addArc arc False (pot False True)  true
        , addArc arc True  (pot True  False) false
        , addArc arc False (pot False False) false ]


----------------------------------------------
-- Constrained decoding'
----------------------------------------------


-- | Constrained version of `treeTagGlobal`
treeTagConstrained'
  :: Graph a b
  -> M.Map Arc (Vec8 Pot)
  -> M.Map G.Vertex Double
  -> Labeling Bool
treeTagConstrained' graph labMap nodMap =
  let Best4{..} =
        tagSubTreeC'
          (treeRoot graph)
          graph
          (fmap explicate labMap)
          nodMap
      best = List.foldl1' better
        -- NOTE: `falseZeroOne` can be excluded in constrained decoding
        [true, falseZeroTrue, falseMoreTrue]
   in getAny <$> bestLab best


-- | Calculate `Best3` of the subtree.
tagSubTreeC'
  :: G.Vertex
    -- ^ Root of the subtree
  -> Graph a b
    -- ^ The entire graph (tree)
  -> M.Map Arc (M.Map (Out Bool) Double)
    -- ^ Explicated labeling potentials
  -> M.Map G.Vertex Double
    -- ^ Node labeling potentials
  -> Best4
tagSubTreeC' root graph lmap nmap =
  List.foldl' (<>)
    (emptyBest4 root $ nmap M.! root)
    (map bestFor children)
  where
    children = Graph.incoming root graph
    bestFor child =
      let arc = (child, root)
          pot arcv hedv depv = (lmap M.! arc) M.!
            Out {arcVal=arcv, hedVal=hedv, depVal=depv}
          Best4{..} = tagSubTreeC' child graph lmap nmap
          -- NOTE: some of the configurations below are not allowed in
          -- constrained decoding and hence are commented out.
          true' = List.foldl1' better
            [ addArc arc True  (pot True  True True)  true
            , addArc arc False (pot False True True)  true
            -- , addArc arc True  (pot True  True False) falseZeroTrue
            , addArc arc False (pot False True False) falseZeroTrue
            , addArc arc True  (pot True  True False) falseOneTrue
            -- , addArc arc False (pot False True False) falseOneTrue
            , addArc arc True  (pot True  True False) falseMoreTrue
            , addArc arc False (pot False True False) falseMoreTrue
            ]
          falseZeroTrue' = List.foldl1' better
            [ addArc arc False (pot False False True)  true
            , addArc arc False (pot False False False) falseZeroTrue
            -- , addArc arc False (pot False False False) falseOneTrue
            , addArc arc False (pot False False False) falseMoreTrue
            ]
          falseOneTrue' = List.foldl1' better
            [ addArc arc True (pot True False True)  true
            -- , addArc arc True (pot True False False) falseZeroTrue
            , addArc arc True (pot True False False) falseOneTrue
            , addArc arc True (pot True False False) falseMoreTrue
            ]
       in Best4
            { true = true'
            , falseZeroTrue = falseZeroTrue'
            , falseOneTrue  = falseOneTrue'
            , falseMoreTrue = impossible
            }


----------------------------------------------
-- Constrained decoding
----------------------------------------------


-- | The best labeling
data Best4 = Best4
  { true          :: Best
    -- ^ The label of the root is `True`.  The root's outgoing arc can be
    -- `True` or `False.
  , falseZeroTrue :: Best
    -- ^ The label of the root is `False` and all its incoming arcs are `False`
    -- too.  The outgoing arc must be `False`.
  , falseOneTrue  :: Best
    -- ^ The label of the root is `False` and exactly one of its incoming arcs
    -- is `True`.  The outgoing arc must be `True`.
  , falseMoreTrue :: Best
    -- ^ The label of the root is `False` and more than one of its incoming
    -- arcs is `True`.  The outgoing arc can be `True` or `False.
  }

instance Semigroup Best4 where
  b1 <> b2 = Best4
    { true =
        true b1 <> true b2
    , falseZeroTrue =
        falseZeroTrue b1 <> falseZeroTrue b2
    , falseOneTrue = List.foldl1' better
        [ falseZeroTrue b1 <> falseOneTrue  b2
        , falseOneTrue  b1 <> falseZeroTrue b2
        ]
    , falseMoreTrue = List.foldl1' better
        [ falseZeroTrue b1 <> falseMoreTrue b2
        , falseMoreTrue b1 <> falseZeroTrue b2
        , falseOneTrue  b1 <> falseOneTrue  b2
        , falseOneTrue  b1 <> falseMoreTrue b2
        , falseMoreTrue b1 <> falseOneTrue  b2
        , falseMoreTrue b1 <> falseMoreTrue b2
        ]
    }

-- instance Monoid Best4 where
--   mempty = Best4 mempty mempty impossible impossible


-- | Empty `Best4` for a given tree node.  Think of `mempty` with obligatory
-- vertex and potential argument.
emptyBest4 :: G.Vertex -> Double -> Best4
emptyBest4 node pot = Best4
  { true = addNode node True pot mempty
  , falseZeroTrue = addNode node False 0.0 mempty
  , falseOneTrue = impossible
  , falseMoreTrue = impossible
  }


-- | Constrained version of `treeTagGlobal`
treeTagConstrained
  :: Graph a b
  -> M.Map Arc (Vec8 Pot)
  -> Labeling Bool
treeTagConstrained graph labMap =
  let Best4{..} =
        tagSubTreeC
          (treeRoot graph)
          graph
          (fmap explicate labMap)
      best = List.foldl1' better
        -- NOTE: `falseZeroOne` can be excluded in constrained decoding
        [true, falseZeroTrue, falseMoreTrue]
   in getAny <$> bestLab best


-- | Calculate `Best3` of the subtree.
tagSubTreeC
  :: G.Vertex
    -- ^ Root of the subtree
  -> Graph a b
    -- ^ The entire graph (tree)
  -> M.Map Arc (M.Map (Out Bool) Double)
    -- ^ Explicated labeling potentials
  -> Best4
tagSubTreeC root graph lmap =
  List.foldl' (<>) (emptyBest4 root 0.0) (map bestFor children)
  where
    children = Graph.incoming root graph
    bestFor child =
      let arc = (child, root)
          pot arcv hedv depv = (lmap M.! arc) M.!
            Out {arcVal=arcv, hedVal=hedv, depVal=depv}
          Best4{..} = tagSubTreeC child graph lmap
          -- NOTE: some of the configurations below are not allowed in
          -- constrained decoding and hence are commented out.
          true' = List.foldl1' better
            [ addArc arc True  (pot True  True True)  true
            , addArc arc False (pot False True True)  true
            -- , addArc arc True  (pot True  True False) falseZeroTrue
            , addArc arc False (pot False True False) falseZeroTrue
            , addArc arc True  (pot True  True False) falseOneTrue
            -- , addArc arc False (pot False True False) falseOneTrue
            , addArc arc True  (pot True  True False) falseMoreTrue
            , addArc arc False (pot False True False) falseMoreTrue
            ]
          falseZeroTrue' = List.foldl1' better
            [ addArc arc False (pot False False True)  true
            , addArc arc False (pot False False False) falseZeroTrue
            -- , addArc arc False (pot False False False) falseOneTrue
            , addArc arc False (pot False False False) falseMoreTrue
            ]
          falseOneTrue' = List.foldl1' better
            [ addArc arc True (pot True False True)  true
            -- , addArc arc True (pot True False False) falseZeroTrue
            , addArc arc True (pot True False False) falseOneTrue
            , addArc arc True (pot True False False) falseMoreTrue
            ]
       in Best4
            { true = true'
            , falseZeroTrue = falseZeroTrue'
            , falseOneTrue  = falseOneTrue'
            , falseMoreTrue = impossible
            }


-- ----------------------------------------------
-- -- Experiments with global training
-- ----------------------------------------------
-- 
-- 
-- -- | Calculate the *marginal* probabilities of the individual node/arc
-- -- labelings.  This code is 100% experimental, no idea if it's going to work or
-- -- if the idea is even sound.
-- marginalize
--   :: Graph a b
--   -> M.Map Arc (BVar s (Vec8 Pot))
--   -> M.Map Arc (BVar s (Vec8 Prob))
-- marginalize graph potMap =
--   undefined
-- 
-- 
-- -- -- | Perform the inside pass of the inside-outside algorithm.  On input the
-- -- -- function takes a map of potential vectors.  On output it should give sth.
-- -- -- like inside maginal probabilities...  To clarify.
-- -- inside
-- --   :: Graph a b
-- --   -> M.Map Arc (BVar s (Vec8 Pot))
-- --   -> M.Map Arc (BVar s (Vec8 Prob))
-- -- inside = undefined
-- 
-- 
-- -- | Inside *potentials*.  For the given vertex, the function (recursively)
-- -- calculates the sum of potentials of its possible inside labelings.  It also
-- -- preserves the inside potentials of the descendant vertices.
-- insideSub
--   :: (Reifies s W)
--   => Graph a b
--   -> M.Map Arc (BVar s (Vec8 Pot))
--   -> G.Vertex
--   -> M.Map (G.Vertex, Bool) (BVar s Double)
-- insideSub graph potMap =
-- 
--   -- TODO: consider using memoization to calculate a function from (G.Vertex,
--   -- Bool) to (BVar s Double) rather than storing the result in the map.
--   undefined -- go
-- 
--   where
-- 
--     -- Recursively calculate the inside potential for the given node and label.
--     insidePot w wMwe = sum $ do
--       -- Loop over all @w@'s children nodes
--       v <- Graph.incoming w graph
--       -- Pick the MWE label of the child @v@
--       vMwe <- [False, True]
--       -- Pick the MWE label of the arc connecting @v@ with @w@; note that it is
--       -- possible to calculate the resulting potential faster, without
--       -- enumerating the possible arc labels
--       arcMwe <- [False, True]
--       -- The resulting potential
--       return $ insidePot v vMwe + arcPot potMap (v, vMwe) (w, wMwe) arcMwe


-- -- | Calculates the arc potential given the label of the head and the label of
-- -- the dependent.
-- arcPot 
--   :: forall s. (Reifies s W)
--   => M.Map Arc (BVar s (Vec8 Pot))
--     -- ^ Potential map (as calculated with the network)
--   -> (G.Vertex, Bool) 
--     -- ^ Dependent node and its label
--   -> (G.Vertex, Bool)
--     -- ^ Head node and its label
--   -> Bool
--     -- ^ Label of the arc
--   -> BVar s Double
-- arcPot potMap (dep, depMwe) (hed, hedMwe) arcMwe =
--   potVec `dot` BP.constVar (mask out)
--   where
--     potVec = BP.coerceVar (potMap M.! (dep, hed))
--     out = Out
--       { arcVal = arcMwe
--       , depVal = depMwe
--       , hedVal = hedMwe
--       }


----------------------------------------------
-- Explicating
----------------------------------------------


-- | Determine the values assigned to different labellings of the given arc and
-- nodes.
explicate :: Vec8 p -> M.Map (Out Bool) Double
explicate = M.fromList . zip B.enumerate . toList . unVec


-- | The inverse of `explicate`.
obfuscate :: M.Map (Out Bool) Double -> Vec8 p
obfuscate = Vec . LA.vector . M.elems


----------------------------------------------
-- Serialization
----------------------------------------------


-- | Save the parameters in the given file.
save :: (Binary a) => FilePath -> a -> IO ()
save path =
  BL.writeFile path . compress . Bin.encode


-- | Load the parameters from the given file.
load :: (Binary a) => FilePath -> IO a
load path =
  Bin.decode . decompress <$> BL.readFile path


----------------------------------------------
-- DataSet
----------------------------------------------


-- | Dataset element
data Elem a = Elem
  { graph :: Graph a ()
    -- ^ Input graph
  , arcMap :: M.Map Arc Double
    -- ^ Target arc probabilities
  , nodMap :: M.Map G.Vertex Double
    -- ^ Target node probabilities
  , tokMap :: M.Map G.Vertex Cupt.Token
    -- ^ Map of .cupt tokens
  } deriving (Show, Generic, Binary)

instance Functor Elem where
  fmap f el = el {graph = nmap f (graph el)}


-- | Get the list of tokens with the corresponding labels present in the given
-- training element.
tokens :: Elem a -> [(Cupt.Token, a)]
tokens el = do
  (v, tok) <- M.toList (tokMap el)
  let y = just $ nodeLabAt (graph el) v
  return (tok, y)
  where
    just Nothing = error "Neg.Graph2.tokens: got Nothing"
    just (Just x) = x


-- | Replace the labels in the given training element.
replace :: [b] -> Elem a -> Elem b
replace xs el =
  el {graph = nmap' update (graph el)}
  where
    update v _ = newMap M.! v
    newMap = M.fromList $ do
      ((v, _tok), x) <- zip (M.toList $ tokMap el) xs
      return (v, x)


----------------------------------------------
-- Error
----------------------------------------------


-- | Cross entropy between the true and the artificial distributions
crossEntropyHard
  :: forall s. (Reifies s W)
  => BVar s (Vec8 Prob)
    -- ^ Target ,,true'' distribution
  -> BVar s (Vec8 Prob)
    -- ^ Output ,,artificial'' distribution
  -> BVar s Double
crossEntropyHard p0 q0 =
  negate (p `dot` LBP.vmap' log' q)
  where
    p = BP.coerceVar p0 :: BVar s (R 8)
    q = BP.coerceVar q0
    -- avoid NaN when p = 0 and q = 0
    log' x
      | x > 0 = log x
      | otherwise = -1.0e8


-- -- | Cross entropy between the true and the artificial distributions
-- crossEntropyHard
--   :: forall s. (Reifies s W)
--   => BVar s (Vec8 Prob)
--     -- ^ Target ,,true'' distribution
--   -> BVar s (Vec8 Prob)
--     -- ^ Output ,,artificial'' distribution
--   -> BVar s Double
-- crossEntropyHard p0 q0
--   | vp `at` 0 > 0 && vq `at` 0 <= 0 = error "1"
--   | vp `at` 1 > 0 && vq `at` 1 <= 0 = error "2"
--   | vp `at` 2 > 0 && vq `at` 2 <= 0 = error "3"
--   | vp `at` 3 > 0 && vq `at` 3 <= 0 = error "4"
--   | vp `at` 4 > 0 && vq `at` 4 <= 0 = error "5"
--   | vp `at` 5 > 0 && vq `at` 5 <= 0 = error "6"
--   | vp `at` 6 > 0 && vq `at` 6 <= 0 = error "7"
--   | vp `at` 7 > 0 && vq `at` 7 <= 0 = error "8"
--   | vq `at` 0 <= 0 = error "1.2"
--   | vq `at` 1 <= 0 = error "2.2"
--   | vq `at` 2 <= 0 = error "3.2"
--   | vq `at` 3 <= 0 = error "4.2"
--   | vq `at` 4 <= 0 = error (show $ map (\x -> vp `at` 4 > BP.auto x) [-0.1, 0.0 .. 1.0])
--   | vq `at` 5 <= 0 = error "6.2"
--   | vq `at` 6 <= 0 = error "7.2"
--   | vq `at` 7 <= 0 = error "8.2"
--   | otherwise = result
--   where
--     result = negate (p `dot` LBP.vmap' log q)
--     p = BP.coerceVar p0 :: BVar s (R 8)
--     q = BP.coerceVar q0
-- 
--     vp = LBP.extractV p
--     vq = LBP.extractV q
-- 
--     at v k = maybe 0 id $ v ^^? ix k


-- -- | Cross-entropy between the true and the artificial distributions
-- crossEntropyOne
--   :: (Reifies s W)
--   => BVar s Double
--     -- ^ Target ,,true'' MWE probability
--   -> BVar s Double
--     -- ^ Output ,,artificial'' MWE probability
--   -> BVar s Double
-- crossEntropyOne p q = negate
--   ( p0 * log q0
--   + p1 * log q1
--   )
--   where
--     p1 = p
--     p0 = 1 - p1
--     q1 = q
--     q0 = 1 - q1
-- 
-- 
-- -- | Soft version of `crossEntropy`
-- crossEntropySoft
--   :: forall s. (Reifies s W)
--   => BVar s (Vec8 Prob)
--     -- ^ Target ,,true'' distribution
--   -> BVar s (Vec8 Prob)
--     -- ^ Output ,,artificial'' distribution
--   -> BVar s Double
-- crossEntropySoft p q
--   = sum
--   . map (uncurry crossEntropyOne)
--   $ zip (flatten p) (flatten q)
--   where
--     flatten = F.toList . B.squash


-- | Selected version of cross-entropy.
crossEntropy
  :: forall s. (Reifies s W)
  => BVar s (Vec8 Prob)
    -- ^ Target ,,true'' distribution
  -> BVar s (Vec8 Prob)
    -- ^ Output ,,artificial'' distribution
  -> BVar s Double
crossEntropy =
  -- crossEntropySoft
  crossEntropyHard
{-# crossEntropy #-}


-- | Cross entropy between the target and the output values
errorOne
  :: (Ord a, Reifies s W)
  => M.Map a (BVar s (Vec8 Prob))
    -- ^ Target values
  -> M.Map a (BVar s (Vec8 Prob))
    -- ^ Output values
  -> BVar s Double
errorOne target output = sum $ do
  (key, tval) <- M.toList target
  let oval = output M.! key
  return $ crossEntropy tval oval


-- | Error on a dataset.
errorMany
  :: (Ord a, Reifies s W)
  => [M.Map a (BVar s (Vec8 Prob))] -- ^ Targets
  -> [M.Map a (BVar s (Vec8 Prob))] -- ^ Outputs
--   => [M.Map a (BVar s (R n))] -- ^ Targets
--   -> [M.Map a (BVar s (R n))] -- ^ Outputs
-- --   => [M.Map a (BVar s Double)] -- ^ Targets
-- --   -> [M.Map a (BVar s Double)] -- ^ Outputs
  -> BVar s Double
errorMany targets outputs =
  go targets outputs
  where
    go ts os =
      case (ts, os) of
        (t:tr, o:or) -> errorOne t o + go tr or
        ([], []) -> 0
        _ -> error "errorMany: lists of different size"


-- | Network error on a given dataset.
netError
  :: ( Reifies s W, KnownNat dim
     , B.BiComp dim comp
     )
  => ProbTyp
  -> [Elem (BVar s (R dim))]
  -> BVar s comp
  -> BVar s Double
netError ptyp dataSet net =
  let
    inputs = map graph dataSet
    outputs = map (run ptyp net) inputs
    -- targets = map (fmap BP.auto . mkTarget) dataSet
    targets = map mkTarget dataSet
  in
    errorMany targets outputs


-- | Network error on a given dataset.
netError'
  :: ( Reifies s W, KnownNat dim
     , B.BiComp dim comp
     , U.UniComp dim comp'
     )
  => ProbTyp
  -> [Elem (BVar s (R dim))]
  -> BVar s comp
  -> BVar s comp'
  -> BVar s Double
netError' ptyp dataSet net netU =
  let
    inputs = map graph dataSet
    outputs = map (runBoth ptyp net netU) inputs
    -- outputs = map (run ptyp net) inputs
    targets = map mkTarget dataSet
  in
    errorMany targets outputs


-- | Create the target map from the given dataset element.
mkTarget
  :: (Reifies s W)
  => Elem (BVar s a)
  -> M.Map Arc (BVar s (Vec8 Prob))
mkTarget el = fmap BP.auto . M.fromList $ do
  ((v, w), arcP) <- M.toList (arcMap el)
  let vP = nodMap el M.! v
      wP = nodMap el M.! w
      target = Out
        { arcVal = arcP
        , hedVal = wP
        , depVal = vP }
  return ((v, w), encode target)


-- -- | A version of `netError working on transformed embeddings
-- netError'
--   :: ( Reifies s W, KnownNat d
--      , Ord a, Show a, Ord b, Show b
--      , B.BiComp d a b comp
--      )
--   => ProbTyp
-- --   -> Proxy d
-- --     -- ^ TODO: remove; we don't actually need @d@!
--   -> [(Elem d a b, [BVar s (R i)])]
--     -- ^ Input elements + transformed embeddings
--   -> BVar s comp
--   -> BVar s Double
-- netError' ptyp dataSet net =
--   let
--     inputs = map graph (map fst dataSet)
--     outputs = map (run ptyp net) inputs
--     targets = map (fmap BP.auto . mkTarget) (map fst dataSet)
--   in
--     errorMany targets outputs


----------------------------------------------
-- Encoding
----------------------------------------------


-- | Encode the output structure *with probabilities* as a vector.
--
-- TODO: Out Double -> e.g. Out (Real Prob) or Out (Float Prob)
--
encode :: Out Double -> Vec8 Prob
encode Out{..} = (Vec . encode') [arcVal, hedVal, depVal]


-- | Decode the output structure from the given probability vector.  This
-- function is potentially lossy in the sense that `Vec8 Prob` encodes a joint
-- distibution and the resulting `Out Double` encodes three distributions
-- assumed to be independent.
decode :: Vec8 Prob -> Out Double
decode = BP.evalBP $ BP.collectVar . B.squash
--   case decode' (unVec vec) of
--     [arcP, hedP, depP] -> Out
--       { arcVal = arcP
--       , hedVal = hedP
--       , depVal = depP
--       }
--     xs -> error $
--       "Graph2.decode: unsupported list length (" ++
--        show xs ++ ")"


-- | Encode a list of probabilities of length @n@ as a vector of length @2^n@.
encode'
  :: (KnownNat n)
  => [Double]
  -> R n
encode' =
  LA.vector . go
  where
    go (p:ps)
      = map (uncurry (*))
      $ cartesian [1-p, p] (go ps)
    go [] = [1]


-- -- | Decode the list of probabilities/potentials from the given vector.
-- decode'
--   :: (KnownNat n)
--   => R n
--   -> [Double]
-- decode' =
--   go . toList
--   where
--     go []  = []
--     go [_] = []
--     go xs =
--       let (left, right) = rightInTwo xs
--           -- p0 = sum left
--           p1 = sum right
--        in p1 : go (map (uncurry (+)) (zip left right))


-- | Cartesian product of two lists
cartesian :: [a] -> [b] -> [(a, b)]
cartesian xs ys = do
  x <- xs
  y <- ys
  return (x, y)


-- | Split a list x_1, x_2, ..., x_(2n) in two equal parts:
-- 
--   * x_1, x_2, ..., x_n, and
--   * x_(n+1), x_(n+2), ..., x_(2n)
--
rightInTwo :: [a] -> ([a], [a])
rightInTwo xs =
  List.splitAt (length xs `div` 2) xs


----------------------------------------------
-- Tree functions on graphs
--
-- TODO: perhaps some of the functions in
-- this section could be easily tested?
----------------------------------------------
    
    
-- | Retrieve the root of the given directed graph/tree.  It is assumed that
-- each arc points in the direction of the parent in the tree.  The function
-- raises an error if the input graph is not a well-formed tree.
treeRoot :: Graph a b -> G.Vertex
treeRoot g =
  case roots of
    [v] -> v
    [] -> error "Graph2.treeRoot: no root found!"
    _ -> error "Graph2.treeRoot: several roots found!"
  where
    roots = do
      v <- Graph.graphNodes g
      guard . null $ Graph.outgoing v g
      return v


-- | Determine the set of arcs that allows to connect the two vertices.
treeConnectAll
  :: Graph.Graph a b
  -> S.Set G.Vertex
  -> S.Set Graph.Arc
treeConnectAll graph =
  go . S.toList
  where
    go (v:w:us) = treeConnect graph v w `S.union` go (w:us)
    go _ = S.empty


-- | Determine the set of arcs that allows to connect the two vertices.
treeConnect
  :: Graph.Graph a b
  -> G.Vertex
  -> G.Vertex
  -> S.Set Graph.Arc
treeConnect graph v w =
  arcSet v u `S.union` arcSet w u
  where
    arcSet x y = (S.fromList . pathAsArcs) (treePath graph x y)
    u = treeCommonAncestor graph v w


-- | Find tha path from the first to the second vertex.
treePath :: Graph.Graph a b -> G.Vertex -> G.Vertex -> [G.Vertex]
treePath graph v w =
  List.takeWhile (/=w) (treePathToRoot graph v) ++ [w]


-- | Convert the list of vertices to a list of arcs on the path.
pathAsArcs :: [G.Vertex] -> [Graph.Arc]
pathAsArcs (x:y:xs) = (x, y) : pathAsArcs (y:xs)
pathAsArcs _ = []


-- | Commmon ancestor of the two given nodes
treeCommonAncestor
  :: Graph.Graph a b
  -> G.Vertex
  -> G.Vertex
  -> G.Vertex
treeCommonAncestor graph v w =
  firstIntersection
    (treePathToRoot graph v)
    (treePathToRoot graph w)


-- | Find the first common element of the given lists.
firstIntersection :: Eq a => [a] -> [a] -> a
firstIntersection xs ys 
  = fst . safeHead . reverse
  . List.takeWhile (uncurry (==))
  $ zip (reverse xs) (reverse ys)
  where
    safeHead (e:es) = e
    safeHead [] =
      error "Graph2.firstIntersection: no intersection found"


-- | Find the path from the given node to the root of the tree.
treePathToRoot :: Graph.Graph a b -> G.Vertex -> [G.Vertex]
treePathToRoot graph =
  go
  where
    go v =
      case Graph.outgoing v graph of
        [] -> [v]
        [w] -> v : go w
        _ -> error "Graph2.treePathToRoot: the given graph is not a tree"
